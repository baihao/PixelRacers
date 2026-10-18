"""End-to-end submission pipeline: assets -> localize (cached) -> 3 options.

Config via SubmissionConfig. Three options map to three cells / CLI subcommands:
  Option 1  motion_only  — SE dead-reckon sampled at two-camera t_last
  Option 2  two_cam      — offline fr+rl se2 fuse (no EKF)
  Option 3  ekf          — SE predict + fr/rl extrapolate -> se2 -> EKF update R(s)

MODE="eval" (training+GT): also writes metrics + cropped GT.
MODE="predict" (test):     submission.csv only.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
for p in (HERE, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import assets_io
import fusion
import laps as lapmod
import metrics as metricsmod
import motion_ekf as mek
import obs_sigma
import plots
import sfm_core as sfm
import sfm_features as feat
import sfm_localize as sloc
from mcap_io import mcap_time_bounds


DEFAULT_MAP_MCAP = HERE / "track-model" / "map.mcap"
EXPECTED_MAP_PREPROCESS = "clahe_highlight_v1"


@dataclass
class SubmissionConfig:
    train_mcap: Optional[str] = None
    test_mcap: Optional[str] = None
    mode: str = "eval"                       # "eval" | "predict"
    laps: Optional[List[int]] = None         # None = all
    windows: Optional[List[Tuple[int, int]]] = None
    laps_json: Optional[str] = None
    cams: Sequence[str] = ("fr", "rl")
    ref_camera: str = "fr"
    backend: str = "sp_nv_nn"
    query_stride: int = 1
    top_k: int = 5
    map_laps: Tuple[int, ...] = (2, 4)
    override_query: bool = False
    sfm_model_dir: str = str(HERE / "sfm-model")
    track_mcap: str = str(DEFAULT_MAP_MCAP)
    sigma_json: str = str(HERE / "error-model" / "obs_error_sigma_vs_s.json")
    noise_json: str = str(HERE / "error-model" / "se_step_residual_noise.json")
    out_root: str = str(HERE / "outputs")
    # Manual map-frame init (predict / validation bags). None → auto.
    init_pose_xyyaw: Optional[Tuple[float, float, float]] = None
    init_pose_t0_ns: Optional[int] = None
    init_pose_probe: bool = False
    init_pose_probe_max_frames: int = 40
    init_pose_json: Optional[str] = None  # default: run_dir/init_pose.json

    @property
    def data_mcap(self) -> str:
        m = self.train_mcap if self.mode == "eval" else self.test_mcap
        if not m:
            raise ValueError(f"mode={self.mode} requires "
                             f"{'train_mcap' if self.mode == 'eval' else 'test_mcap'}")
        return m

    @property
    def has_gt(self) -> bool:
        return self.mode == "eval"

    def map_dir(self, cam: str) -> Path:
        return Path(self.sfm_model_dir) / f"map_{cam}_lap2+4"

    @property
    def run_dir(self) -> Path:
        stem = Path(self.data_mcap).stem
        return Path(self.out_root) / f"{self.mode}_{stem}"


@dataclass
class RunContext:
    cfg: SubmissionConfig
    assets: assets_io.Assets
    calibs: Dict[str, sfm.CamCalib]
    track_line: object
    windows: List[Dict] = field(default_factory=list)
    feats: object = None
    track_outline: Dict = field(default_factory=dict)
    results: Dict = field(default_factory=dict)   # {window_name: {opt: result}}


# ---------------------------------------------------------------------------
# setup / shared
# ---------------------------------------------------------------------------
def verify_sfm_maps(
    cfg: SubmissionConfig,
    *,
    expected_preprocess: str = EXPECTED_MAP_PREPROCESS,
) -> Dict[str, dict]:
    """Confirm bundled maps match the CLAHE+highlight query pipeline."""
    import json

    root = Path(cfg.sfm_model_dir)
    report: Dict[str, dict] = {}
    for cam in cfg.cams:
        map_dir = cfg.map_dir(cam)
        bundle = map_dir / sfm.MAP_BUNDLE_NAME
        meta_path = map_dir / "build_meta.json"
        if not bundle.is_file():
            raise FileNotFoundError(f"missing map bundle: {bundle}")
        meta = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
        prep = (meta.get("preprocess") or {}).get("version")
        st = bundle.stat()
        report[cam] = {
            "map_dir": str(map_dir.resolve()),
            "bundle_bytes": int(st.st_size),
            "bundle_mtime_ns": int(st.st_mtime_ns),
            "preprocess_version": prep,
            "clahe_ok": prep == expected_preprocess,
        }
        if prep != expected_preprocess:
            raise RuntimeError(
                f"map_{cam}_lap2+4 preprocess={prep!r}, expected "
                f"{expected_preprocess!r}. Re-copy from "
                f"3d-map-test/outputs_hfnet_clahe_fr_rl/ or rebuild with "
                f"sfm_hfnet_fr_rl_clahe_lap3_loc.ipynb"
            )
    return report


def ensure_assets(cfg: SubmissionConfig) -> RunContext:
    """Extract mcap assets, load calibs + racing line, resolve windows."""
    map_report = verify_sfm_maps(cfg)
    for cam, info in map_report.items():
        print(
            f"[setup] map_{cam} CLAHE ok={info['clahe_ok']} "
            f"bytes={info['bundle_bytes']} preprocess={info['preprocess_version']}"
        )

    run_dir = cfg.run_dir
    assets_dir = run_dir / "assets"
    assets = assets_io.ensure_assets(cfg.data_mcap, assets_dir, want_gt=cfg.has_gt)

    calibs = {c: sfm.load_camera_calib(c, assets_dir) for c in cfg.cams}

    # racing line (track-model/map.mcap via tools/track_frame)
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    from track_frame import load_lines  # type: ignore

    track_line = load_lines(str(cfg.track_mcap))["track"]
    obs_sigma.load_params(cfg.sigma_json)

    # Only scan the (large) mcap for time bounds when no other window source
    # applies — i.e. the whole-mcap fallback. Explicit windows, an external
    # laps.json, or GT-derived laps avoid the full-bag scan.
    need_bounds = not (
        cfg.windows
        or cfg.laps_json
        or (cfg.mode == "eval" and assets.gt is not None)
    )
    bounds = mcap_time_bounds(cfg.data_mcap, cfg.ref_camera) if need_bounds else (0, 0)
    windows = lapmod.select_windows(
        mode=cfg.mode,
        gt=assets.gt,
        mcap_bounds=bounds,
        laps=cfg.laps,
        windows=cfg.windows,
        laps_json=Path(cfg.laps_json) if cfg.laps_json else None,
    )
    outline = plots.resolve_track_outline(
        assets_dir / "track_bounds.json",
        Path(cfg.track_mcap),
        track_line=track_line,
    )

    print(f"[setup] mode={cfg.mode} windows={[w['name'] for w in windows]}")
    print(
        f"[setup] cams={tuple(cfg.cams)} backend={cfg.backend} has_gt={cfg.has_gt} "
        f"motion={assets.motion_source} se_rows={len(assets.se)} "
        f"se_pose={assets.se_has_pose}"
    )
    print(
        f"[setup] track_model={cfg.track_mcap} lap={track_line.lap:.1f}m "
        f"outline={list(outline.keys())}"
    )
    return RunContext(
        cfg=cfg,
        assets=assets,
        calibs=calibs,
        track_line=track_line,
        windows=windows,
        track_outline=outline,
    )


def _ensure_feats(ctx: RunContext):
    if ctx.feats is None:
        backend = feat.resolve_backend(ctx.cfg.backend)
        print(f"[setup] loading feature backend: {backend}")
        ctx.feats = feat.make_features(backend)
    return ctx.feats


def load_loc_tables_loose(
    ctx: RunContext, window: Dict, *, warn: bool = True
) -> Optional[Dict[str, pd.DataFrame]]:
    """Load errors.csv if present (ignore fingerprint) — init-only fallback."""
    cfg = ctx.cfg
    loc_root = cfg.run_dir / "loc" / window["name"]
    tables: Dict[str, pd.DataFrame] = {}
    for cam in cfg.cams:
        err = loc_root / f"loc_{cam}" / sloc.ERRORS_NAME
        if not err.is_file():
            return None
        tables[cam] = pd.read_csv(err)
    if warn:
        print(
            f"[loc] loose load window={window['name']} (cache meta stale/missing; "
            f"init-only — re-run Option 2 with OVERRIDE_QUERY=True for valid query)"
        )
    return tables


def load_loc_tables_cache(
    ctx: RunContext, window: Dict
) -> Optional[Dict[str, pd.DataFrame]]:
    """Load per-camera query results from disk if cache is valid (no feature backend)."""
    cfg = ctx.cfg
    loc_root = cfg.run_dir / "loc" / window["name"]
    backend = feat.resolve_backend(cfg.backend)
    win = (int(window["start_ns"]), int(window["end_ns"]))
    mcap = Path(cfg.data_mcap)
    tables: Dict[str, pd.DataFrame] = {}
    for cam in cfg.cams:
        df = sloc.load_cached_errors(
            cam,
            mcap=mcap,
            window=win,
            map_dir=cfg.map_dir(cam),
            out_loc_dir=loc_root / f"loc_{cam}",
            backend=backend,
            query_stride=cfg.query_stride,
            top_k=cfg.top_k,
        )
        if df is None:
            return None
        tables[cam] = df
    print(f"[loc] cache hit all cams window={window['name']} (no re-query)")
    return tables


def ensure_loc_tables(ctx: RunContext, window: Dict) -> Dict[str, pd.DataFrame]:
    """Online localize (or cache hit) for each camera over the window."""
    cfg = ctx.cfg
    if not cfg.override_query:
        cached = load_loc_tables_cache(ctx, window)
        if cached is not None:
            return cached
    feats = _ensure_feats(ctx)
    loc_root = cfg.run_dir / "loc" / window["name"]
    tables: Dict[str, pd.DataFrame] = {}
    for cam in cfg.cams:
        df = sloc.localize_camera(
            cam,
            mcap=Path(cfg.data_mcap),
            window=(window["start_ns"], window["end_ns"]),
            map_dir=cfg.map_dir(cam),
            out_loc_dir=loc_root / f"loc_{cam}",
            feats=feats,
            calib=ctx.calibs[cam],
            gt=ctx.assets.gt,
            backend=feat.resolve_backend(cfg.backend),
            query_stride=cfg.query_stride,
            top_k=cfg.top_k,
            override_query=cfg.override_query,
        )
        tables[cam] = df
    return tables


def _window_se(ctx: RunContext, window: Dict) -> pd.DataFrame:
    se = ctx.assets.se
    m = (se.timestamp_ns >= window["start_ns"]) & (se.timestamp_ns <= window["end_ns"])
    return se[m].reset_index(drop=True)


INIT_POSE_FILENAME = "init_pose.json"


def init_pose_json_path(cfg: SubmissionConfig) -> Path:
    if cfg.init_pose_json:
        return Path(cfg.init_pose_json)
    return cfg.run_dir / INIT_POSE_FILENAME


def load_init_pose_record(cfg: SubmissionConfig, window_name: str) -> Optional[Dict]:
    path = init_pose_json_path(cfg)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    rec = data.get("windows", {}).get(window_name)
    if rec is None:
        return None
    for k in ("x", "y", "yaw"):
        if k not in rec:
            return None
    return rec


def save_init_pose_record(
    cfg: SubmissionConfig,
    window_name: str,
    *,
    x: float,
    y: float,
    yaw: float,
    t0_ns: int,
    source: str,
    **extra,
) -> Path:
    path = init_pose_json_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: Dict = {"mcap": str(cfg.data_mcap), "windows": {}}
    if path.is_file():
        data = json.loads(path.read_text())
    data["mcap"] = str(cfg.data_mcap)
    data.setdefault("windows", {})[window_name] = {
        "x": float(x),
        "y": float(y),
        "yaw": float(yaw),
        "t0_ns": int(t0_ns),
        "source": source,
        **extra,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[init_pose] saved {window_name} -> {path}")
    return path


def probe_init_pose(
    ctx: RunContext,
    window: Dict,
    *,
    max_frames: Optional[int] = None,
) -> Optional[Tuple[float, float, float, int]]:
    """Lightweight PnP on first frames (one camera) for approximate init pose."""
    cfg = ctx.cfg
    if not cfg.init_pose_probe:
        return None
    nmax = int(max_frames or cfg.init_pose_probe_max_frames)
    win = (int(window["start_ns"]), int(window["end_ns"]))
    feats = _ensure_feats(ctx)
    for cam in (cfg.ref_camera, *cfg.cams):
        if cam not in ctx.calibs:
            continue
        hit = sloc.probe_first_visual_pose(
            cam,
            mcap=Path(cfg.data_mcap),
            window=win,
            map_dir=cfg.map_dir(cam),
            feats=feats,
            calib=ctx.calibs[cam],
            gt=ctx.assets.gt,
            top_k=cfg.top_k,
            max_frames=nmax,
        )
        if hit is not None:
            return hit
    return None


def _init_pose_from_visual(
    ctx: RunContext,
    window: Dict,
    se_win: pd.DataFrame,
    *,
    loc_tables: Optional[Dict[str, pd.DataFrame]] = None,
    run_localize: bool = True,
) -> Tuple[float, float, float, int]:
    """Test bags: validation twist/accel have no pose — anchor from first visual fix."""
    t_se0 = int(se_win.iloc[0].timestamp_ns)
    if loc_tables is None:
        loc_tables = load_loc_tables_cache(ctx, window)
    if loc_tables is None and not run_localize:
        loc_tables = load_loc_tables_loose(ctx, window)
    if loc_tables is None and not run_localize and ctx.cfg.init_pose_probe:
        hit = probe_init_pose(ctx, window)
        if hit is not None:
            x, y, yaw, stamp = hit
            print(f"[init_pose] probe -> use stamp={stamp} (SE t0={t_se0})")
            return x, y, yaw, t_se0
    if loc_tables is None:
        if not run_localize:
            raise RuntimeError(
                "No init pose: run run_init_pose.ipynb (probe / manual → init_pose.json), "
                "or Option 2/3 loc cache, or set cfg.init_pose_xyyaw."
            )
        loc_tables = ensure_loc_tables(ctx, window)
    for cam in (ctx.cfg.ref_camera, *ctx.cfg.cams):
        if cam not in loc_tables:
            continue
        df = loc_tables[cam].sort_values("stamp_ns")
        ok = df[df.ok.astype(bool) & (df.stamp_ns >= t_se0)]
        if len(ok) == 0:
            ok = df[df.ok.astype(bool)]
        if len(ok) == 0:
            continue
        r = ok.iloc[0]
        print(
            f"[init_pose] validation-only: use first {cam} visual fix "
            f"stamp={int(r.stamp_ns)}"
        )
        return float(r.pred_x), float(r.pred_y), float(r.pred_yaw), t_se0
    if not run_localize and ctx.cfg.init_pose_probe:
        hit = probe_init_pose(ctx, window)
        if hit is not None:
            x, y, yaw, stamp = hit
            print(f"[init_pose] probe (no ok in cache) stamp={stamp} SE t0={t_se0}")
            return x, y, yaw, t_se0
    raise RuntimeError(
        "No visual localization fix for init pose (validation-only test mcap). "
        "Run run_init_pose.ipynb or two_cam/ekf localize (loc cache)."
    )


def _init_pose(
    ctx: RunContext,
    se_win: pd.DataFrame,
    window: Dict,
    *,
    loc_tables: Optional[Dict[str, pd.DataFrame]] = None,
    run_localize: bool = True,
) -> Tuple[float, float, float, int]:
    """First-frame init: GT (eval), SE pose (train), or first visual fix (test validation)."""
    t_se0 = int(se_win.iloc[0].timestamp_ns)
    manual = ctx.cfg.mode != "eval"
    if manual and ctx.cfg.init_pose_t0_ns is not None:
        t_se0 = int(ctx.cfg.init_pose_t0_ns)
    if ctx.assets.has_gt:
        gt = ctx.assets.gt
        g = gt.iloc[(gt.timestamp_ns - t_se0).abs().argmin()]
        return float(g.x), float(g.y), float(g.yaw), t_se0
    if manual and ctx.cfg.init_pose_xyyaw is not None:
        x, y, yaw = ctx.cfg.init_pose_xyyaw
        print(f"[init_pose] manual xyyaw=({x:.3f},{y:.3f},{yaw:.3f}) t0={t_se0}")
        return float(x), float(y), float(yaw), t_se0
    rec = load_init_pose_record(ctx.cfg, window["name"])
    if rec is not None:
        if rec.get("t0_ns") is not None:
            t_se0 = int(rec["t0_ns"])
        print(
            f"[init_pose] json ({rec.get('source', '?')}) "
            f"xyyaw=({rec['x']:.3f},{rec['y']:.3f},{rec['yaw']:.3f}) t0={t_se0} "
            f"path={init_pose_json_path(ctx.cfg)}"
        )
        return float(rec["x"]), float(rec["y"]), float(rec["yaw"]), t_se0
    if ctx.assets.se_has_pose:
        r0 = se_win.iloc[0]
        return float(r0.x), float(r0.y), float(r0.yaw), int(r0.timestamp_ns)
    return _init_pose_from_visual(
        ctx, window, se_win, loc_tables=loc_tables, run_localize=run_localize
    )


def _finalize(
    ctx: RunContext,
    pred: pd.DataFrame,
    out_dir: Path,
    name: str,
    window: Dict,
    *,
    n_total: Optional[int] = None,
    show_plots: bool = True,
) -> dict:
    """Write submission.csv (+ metrics in eval), draw per-option figure.

    ``pred`` holds only the emitted (successful) poses. ``n_total`` is the
    number of query frames the option was offered (for two_cam, the count of
    sync keys; for motion_only/ekf, equal to len(pred) since they always emit).
    The official ``success_rate_%`` is computed against n_total, so two_cam no
    longer reports a misleading 100%.

    Returns {name, window, pred, gt_at, metrics, out_dir, fig_path}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mek.write_submission(pred, out_dir / "submission.csv")

    n_total = int(n_total) if n_total is not None else len(pred)
    n_emit = int(len(pred))

    gt_at = None
    met: Optional[dict] = None
    if ctx.assets.has_gt:
        gt_at = mek.interp_gt(ctx.assets.gt, pred.timestamp_ns.to_numpy())
        met = metricsmod.compute_camloc_metrics(
            stamps_ns=pred.timestamp_ns.to_numpy(),
            gt_xy=np.column_stack([gt_at.x, gt_at.y]),
            gt_yaw=gt_at.yaw.to_numpy(),
            pred_xy=np.column_stack([pred.x, pred.y]),
            pred_yaw=pred.yaw.to_numpy(),
            ok=np.ones(len(pred), dtype=bool),
            map_mcap=Path(ctx.cfg.track_mcap),
        )
        # Error metrics (RMSE etc.) are over emitted poses; but success rate +
        # within-1m-of-all must use the true query total (n_total), so dropped
        # frames count against two_cam rather than vanishing.
        met["mode"] = name
        met["n_query"] = n_total
        met["n_success"] = n_emit
        met["success_rate_%"] = 100.0 * n_emit / max(n_total, 1)
        if "within_1m_%" in met and np.isfinite(met.get("within_1m_%", np.nan)):
            within_cnt = met["within_1m_%"] / 100.0 * n_emit
            met["within_1m_of_all_%"] = 100.0 * within_cnt / max(n_total, 1)
        pd.Series(met).to_csv(out_dir / "metrics_official.csv", header=["value"])
        print(f"[{name}] wrote {out_dir/'submission.csv'} emit={n_emit}/{n_total} "
              f"({met['success_rate_%']:.1f}%) pos_rmse_xy={met.get('pos_rmse_xy_m'):.3f}m")
    else:
        print(f"[{name}] wrote {out_dir/'submission.csv'} emit={n_emit}/{n_total} (predict: no metrics)")

    fig_path = out_dir / "figure.png"
    plots.plot_option(
        name=name,
        pred=pred,
        gt_at=gt_at,
        metrics=met,
        outline=ctx.track_outline,
        out_path=fig_path,
        show=show_plots,
    )

    result = {
        "name": name,
        "window": window["name"],
        "pred": pred,
        "gt_at": gt_at,
        "metrics": met,
        "out_dir": out_dir,
        "fig_path": fig_path,
    }
    ctx.results.setdefault(window["name"], {})[name] = result
    return result


# ---------------------------------------------------------------------------
# Option 1 / 2 / 3
# ---------------------------------------------------------------------------
def _camera_tlast(ctx: RunContext, window: Dict) -> np.ndarray:
    """Two-camera t_last timeline from camera STAMPS only (no localize).

    Reads header stamps straight from the mcap (no JPEG decode, no features),
    so motion-only never touches the camera image pipeline.
    """
    from mcap_io import camera_stamps

    cfg = ctx.cfg
    stamps = {
        c: np.asarray(
            camera_stamps(
                cfg.data_mcap, c, start_ns=window["start_ns"], end_ns=window["end_ns"]
            ),
            dtype=np.int64,
        )
        for c in cfg.cams
    }
    return mek.build_sync_tlast_from_stamps(
        stamps, ref_camera=cfg.ref_camera, cams=cfg.cams
    )


# Public aliases so the notebook can compose the option flows transparently.
window_se = _window_se
init_pose = _init_pose
probe_init_pose = probe_init_pose
load_init_pose_record = load_init_pose_record
save_init_pose_record = save_init_pose_record
init_pose_json_path = init_pose_json_path
camera_tlast = _camera_tlast
finalize = _finalize


def run_motion_only(ctx: RunContext, window: Dict, *, show_plots: bool = True) -> Optional[dict]:
    """Option 1 — motion prediction only (SE dead-reckon, no camera features).

    1. slice SE (state_estimation) rows to the window
    2. init pose: GT first pose (eval) or SE first pose (predict)
    3. two-camera t_last timeline from camera STAMPS only (no PnP)
    4. dead-reckon vx/vy/ax/ay/wz, sample at each t_last
    """
    se_win = _window_se(ctx, window)
    if len(se_win) == 0:
        print(f"[motion_only:{window['name']}] no SE rows in window — skip")
        return None
    x0, y0, yaw0, _ = _init_pose(
        ctx, se_win, window, run_localize=False
    )
    t_last = _camera_tlast(ctx, window)          # stamps only, no features/PnP
    pred = mek.run_mode_a_se_predict_at(se_win, t_last, x0=x0, y0=y0, yaw0=yaw0)
    out_dir = ctx.cfg.run_dir / window["name"] / "motion_only"
    # motion-only always emits one pose per t_last -> n_total == len(pred)
    return _finalize(ctx, pred, out_dir, "motion_only", window,
                     n_total=len(pred), show_plots=show_plots)


def run_two_cam(ctx: RunContext, window: Dict, *, show_plots: bool = True) -> Optional[dict]:
    """Option 2 — two-camera localization (fr+rl se2 fuse, no EKF).

    1. online localize each camera (or cache hit) -> per-cam errors tables
    2. nearest-stamp align fr/rl, omega-weighted SE(2) fuse with gating
    3. emit ONLY frames where fusion succeeded (drops visual failures)
    n_total = all sync keys, so success_rate reflects dropped frames.
    """
    tables = ensure_loc_tables(ctx, window)
    fused = fusion.fuse_loc_tables_se2(tables, ref_camera=ctx.cfg.ref_camera)
    ok = fused[fused.ok].reset_index(drop=True)
    pred = pd.DataFrame(
        {
            "timestamp_ns": ok.stamp_ns.to_numpy(np.int64),
            "x": ok.pred_x.to_numpy(float),
            "y": ok.pred_y.to_numpy(float),
            "yaw": ok.pred_yaw.to_numpy(float),
        }
    )
    out_dir = ctx.cfg.run_dir / window["name"] / "two_cam"
    out_dir.mkdir(parents=True, exist_ok=True)
    fused.to_csv(out_dir / "errors_se2.csv", index=False)
    return _finalize(ctx, pred, out_dir, "two_cam", window,
                     n_total=len(fused), show_plots=show_plots)


def run_ekf(ctx: RunContext, window: Dict, *, show_plots: bool = True) -> Optional[dict]:
    """Option 3 — EKF fusion (SE predict + fr/rl visual update).

    1. online localize each camera (or cache hit)
    2. build sync buckets (fr/rl nearest-stamp, t_last = max stamp)
    3. per bucket: EKF predict to t_last; extrapolate each cam PnP pose to
       t_last; se2-fuse; EKF update with R(s)=diag(sigma(s)^2, .., sigma_yaw^2)
    EKF always emits the predicted pose (even on visual miss) -> n_total==len.
    """
    cfg = ctx.cfg
    tables = ensure_loc_tables(ctx, window)
    se_win = _window_se(ctx, window)
    if len(se_win) == 0:
        print(f"[ekf:{window['name']}] no SE rows in window — skip")
        return None
    noise = mek.write_process_noise_yaml(Path(cfg.noise_json))
    ncfg = mek.NoiseCfg.from_dict(noise)
    x0, y0, yaw0, t0 = _init_pose(ctx, se_win, window, loc_tables=tables)
    buckets = mek.build_sync_buckets(tables, ref_camera=cfg.ref_camera, cams=cfg.cams)
    pred, sim = mek.run_mode_b_ekf(
        se_win,
        buckets,
        ncfg,
        x0=x0,
        y0=y0,
        yaw0=yaw0,
        t0_ns=t0,
        sigma_of_s_fn=obs_sigma.sigma_of_s,
        track_line=ctx.track_line,
        cams=cfg.cams,
    )
    out_dir = cfg.run_dir / window["name"] / "ekf"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sim").mkdir(parents=True, exist_ok=True)
    sim.to_csv(out_dir / "sim" / "ekf_sim.csv", index=False)
    return _finalize(ctx, pred, out_dir, "ekf", window,
                     n_total=len(pred), show_plots=show_plots)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _combined_gt_csv(ctx: RunContext) -> Path:
    cmp_dir = ctx.cfg.run_dir / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    out = cmp_dir / "gt_all.csv"
    parts = []
    gt = ctx.assets.gt
    for w in ctx.windows:
        sl = gt[
            (gt["timestamp_ns"] >= w["start_ns"]) & (gt["timestamp_ns"] <= w["end_ns"])
        ][["timestamp_ns", "x", "y", "yaw"]]
        parts.append(sl)
    pd.concat(parts, ignore_index=True).sort_values("timestamp_ns").to_csv(
        out, index=False
    )
    return out


def _combined_submission_csv(ctx: RunContext, option: str) -> Path:
    cmp_dir = ctx.cfg.run_dir / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    for w in ctx.windows:
        res = ctx.results.get(w["name"], {}).get(option)
        if res is None:
            continue
        p = res["out_dir"] / "submission.csv"
        if p.is_file():
            frames.append(pd.read_csv(p))
        elif len(res["pred"]):
            frames.append(res["pred"])
    if not frames:
        raise ValueError(f"no submission rows to combine for option={option}")
    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp_ns")
        .drop_duplicates(subset=["timestamp_ns"], keep="last")
    )
    out = cmp_dir / f"submission_{option}_all.csv"
    mek.write_submission(df, out)
    return out


def _merged_pred(ctx: RunContext, option: str) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for w in ctx.windows:
        res = ctx.results.get(w["name"], {}).get(option)
        if res is not None and len(res["pred"]):
            parts.append(res["pred"])
    if not parts:
        raise ValueError(f"no predictions in ctx.results for option={option!r}")
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("timestamp_ns")
        .drop_duplicates(subset=["timestamp_ns"], keep="last")
    )


def present_option(
    ctx: RunContext,
    option: str,
    *,
    show_plots: bool = True,
    print_evaluate_report: bool = False,
) -> pd.DataFrame:
    """After an option cell: merged-lap figure + one ``evaluate.py`` table row (window=all).

    Per-lap ``finalize(..., show_plots=False)`` avoids N figures; this shows the combined view.
    """
    import evaluate as evaluate_mod  # noqa: WPS433

    pred = _merged_pred(ctx, option)
    cmp_dir = ctx.cfg.run_dir / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    fig_path = cmp_dir / f"figure_{option}_all.png"

    gt_at = None
    met: Optional[dict] = None
    rows: List[dict] = []

    if ctx.assets.has_gt:
        gt_parts = [
            ctx.assets.gt[
                (ctx.assets.gt["timestamp_ns"] >= w["start_ns"])
                & (ctx.assets.gt["timestamp_ns"] <= w["end_ns"])
            ]
            for w in ctx.windows
        ]
        gt_plot = pd.concat(gt_parts, ignore_index=True).sort_values("timestamp_ns")
        gt_at = mek.interp_gt(gt_plot, pred.timestamp_ns.to_numpy(np.int64))
        met = metricsmod.compute_camloc_metrics(
            stamps_ns=pred.timestamp_ns.to_numpy(),
            gt_xy=np.column_stack([gt_at.x, gt_at.y]),
            gt_yaw=gt_at.yaw.to_numpy(),
            pred_xy=np.column_stack([pred.x, pred.y]),
            pred_yaw=pred.yaw.to_numpy(),
            ok=np.ones(len(pred), dtype=bool),
            map_mcap=Path(ctx.cfg.track_mcap),
        )
        met["mode"] = option
        n_query = 0
        n_success = 0
        for w in ctx.windows:
            r = ctx.results.get(w["name"], {}).get(option)
            if not r:
                continue
            m = r.get("metrics") or {}
            n_query += int(m.get("n_query", len(r["pred"])))
            n_success += int(m.get("n_success", len(r["pred"])))
        met["n_query"] = n_query
        met["n_success"] = n_success
        met["success_rate_%"] = 100.0 * n_success / max(n_query, 1)
        if "within_1m_%" in met and np.isfinite(met.get("within_1m_%", np.nan)):
            within_cnt = met["within_1m_%"] / 100.0 * len(pred)
            met["within_1m_of_all_%"] = 100.0 * within_cnt / max(n_query, 1)

        gt_csv = _combined_gt_csv(ctx)
        sub = _combined_submission_csv(ctx, option)
        m_eval = evaluate_mod.score_submission(str(gt_csv), str(sub))
        uses_vision = option in ("two_cam", "ekf")
        cam_mean, loc_ms_all = _aggregate_loc_timing(ctx)
        row: dict = {"window": "all", "option": option, **m_eval}
        row["loc_ms_mean_fr"] = cam_mean.get("fr") if uses_vision else float("nan")
        row["loc_ms_mean_rl"] = cam_mean.get("rl") if uses_vision else float("nan")
        row["loc_ms_mean_per_image"] = loc_ms_all if uses_vision else float("nan")
        rows.append(row)
        if print_evaluate_report:
            print(f"\n=== tools/evaluate.py — {option} (all laps) ===")
            print(evaluate_mod.format_report(m_eval))

    plots.plot_option(
        name=option,
        pred=pred,
        gt_at=gt_at,
        metrics=met,
        outline=ctx.track_outline,
        out_path=fig_path,
        show=show_plots,
    )
    return pd.DataFrame(rows)


def _aggregate_loc_timing(ctx: RunContext) -> Tuple[Dict[str, float], float]:
    per_cam: Dict[str, List[float]] = {c: [] for c in ctx.cfg.cams}
    all_ms: List[float] = []
    for w in ctx.windows:
        for cam in ctx.cfg.cams:
            p = ctx.cfg.run_dir / "loc" / w["name"] / f"loc_{cam}" / sloc.LOC_TIMES_NAME
            if not p.is_file():
                continue
            ms = (np.load(p).astype(np.float64) * 1000).tolist()
            per_cam[cam].extend(ms)
            all_ms.extend(ms)
    cam_mean = {
        c: float(np.mean(per_cam[c])) if per_cam[c] else float("nan") for c in ctx.cfg.cams
    }
    return cam_mean, float(np.mean(all_ms)) if all_ms else float("nan")


# Columns matching tools/evaluate.py score_submission (+ loc timing).
EVALUATE_TABLE_COLUMNS: Tuple[str, ...] = (
    "window",
    "option",
    "poses_submitted",
    "poses_evaluated",
    "coverage_pct",
    "min_coverage_pct",
    "position_rmse_m",
    "position_median_m",
    "position_p95_m",
    "lateral_rmse_m",
    "longitudinal_rmse_m",
    "yaw_rmse_deg",
    "within_0.25m_pct",
    "within_0.50m_pct",
    "within_1.00m_pct",
    "ranked",
    "loc_ms_mean_fr",
    "loc_ms_mean_rl",
    "loc_ms_mean_per_image",
)

# Row order / labels aligned with tools/evaluate.py format_report (for vertical tables).
EVALUATE_METRIC_ROWS: Tuple[Tuple[str, str], ...] = (
    ("poses_submitted", "poses submitted"),
    ("poses_evaluated", "poses evaluated"),
    ("coverage_pct", "coverage (1 s bins) [%]"),
    ("min_coverage_pct", "min coverage to rank [%]"),
    ("position_rmse_m", "position RMSE [m] (score)"),
    ("position_median_m", "position median [m]"),
    ("position_p95_m", "position p95 [m]"),
    ("lateral_rmse_m", "lateral RMSE [m]"),
    ("longitudinal_rmse_m", "longitudinal RMSE [m]"),
    ("yaw_rmse_deg", "yaw RMSE [deg]"),
    ("within_0.25m_pct", "within 0.25 m [%]"),
    ("within_0.50m_pct", "within 0.50 m [%]"),
    ("within_1.00m_pct", "within 1.00 m [%]"),
    ("ranked", "ranked"),
    ("loc_ms_mean_fr", "loc mean [ms] fr"),
    ("loc_ms_mean_rl", "loc mean [ms] rl"),
    ("loc_ms_mean_per_image", "loc mean [ms] per image"),
)


def _format_eval_cell(key: str, val) -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "—"
    if key in ("poses_submitted", "poses_evaluated"):
        return f"{int(val)}"
    if key == "ranked":
        return "yes" if bool(val) else "no"
    if key.endswith("_pct"):
        return f"{float(val):.1f}"
    if key == "yaw_rmse_deg":
        return f"{float(val):.3f}"
    if key.startswith("loc_ms"):
        return f"{float(val):.1f}"
    if "rmse" in key or "median" in key or "p95" in key or key == "position_rmse_m":
        return f"{float(val):.3f}"
    return str(val)


def evaluate_table_vertical(df: pd.DataFrame) -> pd.DataFrame:
    """Metrics as rows (evaluate.py field order). One option → ``metric`` + ``value``; several → column per option."""
    if df is None or df.empty:
        return df
    rows_def = [(k, lab) for k, lab in EVALUATE_METRIC_ROWS if k in df.columns]
    if len(df) == 1:
        r = df.iloc[0]
        return pd.DataFrame(
            {
                "metric": [lab for _, lab in rows_def],
                "value": [_format_eval_cell(k, r[k]) for k, _ in rows_def],
            }
        )
    by_opt: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        opt = str(r.get("option", "option"))
        by_opt[opt] = [_format_eval_cell(k, r[k]) for k, _ in rows_def]
    out = pd.DataFrame(by_opt, index=[lab for _, lab in rows_def])
    out.index.name = "metric"
    return out


def run_summary(
    ctx: RunContext,
    *,
    options: Sequence[str] = ("motion_only", "two_cam", "ekf"),
    show_plots: bool = True,
    aggregate_laps: bool = True,
) -> Optional[pd.DataFrame]:
    """Combined comparison: metrics table + metrics bars + trajectory overlay.

    eval: writes compare/metrics_compare.csv and comparison figures.
    predict: lists submission rows and draws trajectory overlay only.
    Multi-lap runs aggregate into one ``all`` view (no per-lap summary plots).
    """
    cfg = ctx.cfg
    cmp_dir = cfg.run_dir / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    mdf_out: Optional[pd.DataFrame] = None

    if aggregate_laps and len(ctx.windows) >= 1:
        preds: Dict[str, pd.DataFrame] = {}
        for o in options:
            parts = []
            for w in ctx.windows:
                res = ctx.results.get(w["name"], {})
                if o not in res:
                    continue
                parts.append(res[o]["pred"])
            if parts:
                preds[o] = (
                    pd.concat(parts, ignore_index=True)
                    .sort_values("timestamp_ns")
                    .drop_duplicates(subset=["timestamp_ns"], keep="last")
                )
        if preds:
            gt_parts = [
                ctx.assets.gt[
                    (ctx.assets.gt["timestamp_ns"] >= w["start_ns"])
                    & (ctx.assets.gt["timestamp_ns"] <= w["end_ns"])
                ]
                for w in ctx.windows
            ]
            gt_plot = (
                pd.concat(gt_parts, ignore_index=True).sort_values("timestamp_ns")
                if gt_parts and ctx.assets.has_gt
                else None
            )
            gt_at = None
            if gt_plot is not None and len(gt_plot):
                ref = next(iter(preds.values()))
                gt_at = mek.interp_gt(gt_plot, ref.timestamp_ns.to_numpy(np.int64))
            plots.plot_summary_trajectories(
                preds,
                gt_at=gt_at,
                outline=ctx.track_outline,
                out_path=cmp_dir / "trajectories_all.png",
                title="Summary — trajectories (all laps)",
                show=show_plots,
            )
        if ctx.assets.has_gt:
            etab = build_evaluate_table(ctx, options=options, evaluate=True, per_lap=False)
            bar_rows = []
            for _, r in etab.iterrows():
                bar_rows.append(
                    {
                        "mode": r["option"],
                        "pos_rmse_xy_m": r.get("position_rmse_m"),
                        "pos_rmse_sn_m": float("nan"),
                        "yaw_rmse_deg": r.get("yaw_rmse_deg"),
                        "within_1m_of_all_%": r.get("within_1.00m_pct"),
                    }
                )
            mdf = pd.DataFrame(bar_rows)
            if len(mdf):
                plots.plot_summary_metrics(
                    mdf, out_path=cmp_dir / "metrics_all.png", show=show_plots
                )
                print("\n=== metrics (all laps) ===")
                print(mdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
                mdf_out = mdf
                mdf_out.to_csv(cmp_dir / "metrics_compare.csv", index=False)
        else:
            print("\n=== submissions (all laps) ===")
            for o, pred in preds.items():
                print(f"  {o:>12}  rows={len(pred)}")
        return mdf_out

    for window in ctx.windows:
        wname = window["name"]
        res = ctx.results.get(wname, {})
        present = [o for o in options if o in res]
        if not present:
            continue

        # trajectory overlay (always)
        gt_at = None
        for o in present:
            if res[o]["gt_at"] is not None:
                gt_at = res[o]["gt_at"]
                break
        preds = {o: res[o]["pred"] for o in present}
        plots.plot_summary_trajectories(
            preds,
            gt_at=gt_at,
            outline=ctx.track_outline,
            out_path=cmp_dir / f"trajectories_{wname}.png",
            title=f"Summary — trajectories ({wname})",
            show=show_plots,
        )

        # metrics table + bars (eval only)
        if ctx.assets.has_gt:
            rows = []
            for o in present:
                m = res[o]["metrics"] or {}
                rows.append(
                    {
                        "window": wname,
                        "mode": o,
                        "n_query": m.get("n_query"),
                        "success_rate_%": m.get("success_rate_%"),
                        "pos_rmse_xy_m": m.get("pos_rmse_xy_m"),
                        "pos_rmse_sn_m": m.get("pos_rmse_sn_m"),
                        "yaw_rmse_deg": m.get("yaw_rmse_deg"),
                        "within_1m_of_all_%": m.get("within_1m_of_all_%"),
                    }
                )
            mdf = pd.DataFrame(rows)
            plots.plot_summary_metrics(
                mdf, out_path=cmp_dir / f"metrics_{wname}.png", show=show_plots
            )
            print(f"\n=== metrics ({wname}) ===")
            print(mdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            mdf_out = mdf if mdf_out is None else pd.concat([mdf_out, mdf], ignore_index=True)
        else:
            print(f"\n=== submissions ({wname}) ===")
            for o in present:
                p = res[o]["out_dir"] / "submission.csv"
                print(f"  {o:>12}  rows={len(res[o]['pred'])}  {p}")

    if mdf_out is not None:
        mdf_out.to_csv(cmp_dir / "metrics_compare.csv", index=False)
    return mdf_out


# ---------------------------------------------------------------------------
# submission listing + tools/evaluate.py
# ---------------------------------------------------------------------------
def gt_slice_path(ctx: RunContext, window: Dict) -> Optional[Path]:
    """Write a GT csv cropped to the window (eval only) for tools/evaluate.py."""
    if not ctx.assets.has_gt:
        return None
    out = ctx.cfg.run_dir / window["name"] / f"gt_{window['name']}.csv"
    mek.write_gt_lap_slice(ctx.assets.gt, window["start_ns"], window["end_ns"], out)
    return out


def run_evaluate(gt_csv: Path, submission_csv: Path) -> str:
    """Invoke tools/evaluate.py (official scorer) and return its stdout+stderr."""
    import evaluate as evaluate_mod  # noqa: WPS433  (tools/evaluate.py)

    return evaluate_mod.format_report(
        evaluate_mod.score_submission(str(gt_csv), str(submission_csv))
    )


def loc_timing_stats(ctx: RunContext, window: Dict) -> Dict[str, dict]:
    """Per-camera mean localize time (ms) from cached ``localize_times_s.npy``."""
    loc_root = ctx.cfg.run_dir / "loc" / window["name"]
    stats: Dict[str, dict] = {}
    for cam in ctx.cfg.cams:
        p = loc_root / f"loc_{cam}" / sloc.LOC_TIMES_NAME
        if not p.is_file():
            continue
        t = np.load(p).astype(np.float64)
        stats[cam] = {
            "n_frames": int(len(t)),
            "loc_ms_mean": float(np.mean(t) * 1000) if len(t) else float("nan"),
            "loc_ms_median": float(np.median(t) * 1000) if len(t) else float("nan"),
        }
    return stats


def build_evaluate_table(
    ctx: RunContext,
    *,
    options: Sequence[str] = ("motion_only", "two_cam", "ekf"),
    evaluate: bool = True,
    print_reports: bool = False,
    per_lap: bool = False,
) -> pd.DataFrame:
    """Table aligned with ``tools/evaluate.py`` plus per-image localization time (ms).

    Default ``per_lap=False``: one row per option with ``window=all`` (merged laps).
    """
    import evaluate as evaluate_mod  # noqa: WPS433

    if evaluate and ctx.assets.has_gt and not per_lap:
        gt_csv = _combined_gt_csv(ctx)
        cam_mean, loc_ms_all = _aggregate_loc_timing(ctx)
        rows: List[dict] = []
        for opt in options:
            if not any(opt in ctx.results.get(w["name"], {}) for w in ctx.windows):
                continue
            sub = _combined_submission_csv(ctx, opt)
            m = evaluate_mod.score_submission(str(gt_csv), str(sub))
            uses_vision = opt in ("two_cam", "ekf")
            row: dict = {"window": "all", "option": opt, **m}
            row["loc_ms_mean_fr"] = cam_mean.get("fr") if uses_vision else float("nan")
            row["loc_ms_mean_rl"] = cam_mean.get("rl") if uses_vision else float("nan")
            row["loc_ms_mean_per_image"] = loc_ms_all if uses_vision else float("nan")
            row["combined_submission_csv"] = str(sub)
            row["combined_gt_csv"] = str(gt_csv)
            if print_reports:
                print("=" * 70)
                print(f"[all / {opt}] submission: {sub}")
                print(evaluate_mod.format_report(m))
            rows.append(row)
        if print_reports:
            print("=" * 70)
        df = pd.DataFrame(rows)
        if not df.empty:
            out = ctx.cfg.run_dir / "compare" / "evaluate_table.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
        return df

    rows = []
    for window in ctx.windows:
        wname = window["name"]
        res = ctx.results.get(wname, {})
        gt_csv = gt_slice_path(ctx, window) if ctx.assets.has_gt else None
        loc_stats = loc_timing_stats(ctx, window)
        all_times_ms: List[float] = []
        for cam in ctx.cfg.cams:
            p = ctx.cfg.run_dir / "loc" / wname / f"loc_{cam}" / sloc.LOC_TIMES_NAME
            if p.is_file():
                all_times_ms.extend((np.load(p).astype(np.float64) * 1000).tolist())
        loc_ms_all = float(np.mean(all_times_ms)) if all_times_ms else float("nan")

        for opt in options:
            if opt not in res:
                continue
            sub = res[opt]["out_dir"] / "submission.csv"
            row: dict = {
                "window": wname,
                "option": opt,
                "submission_csv": str(sub),
            }
            uses_vision = opt in ("two_cam", "ekf")
            for cam in ctx.cfg.cams:
                st = loc_stats.get(cam, {})
                row[f"loc_ms_mean_{cam}"] = (
                    st.get("loc_ms_mean") if uses_vision else float("nan")
                )
                row[f"n_loc_frames_{cam}"] = (
                    st.get("n_frames") if uses_vision else float("nan")
                )
            row["loc_ms_mean_per_image"] = loc_ms_all if uses_vision else float("nan")

            if evaluate and ctx.assets.has_gt and gt_csv is not None:
                m = evaluate_mod.score_submission(str(gt_csv), str(sub))
                row.update({**m, "gt_csv": str(gt_csv)})
                if print_reports:
                    print("=" * 70)
                    print(f"[{wname} / {opt}] submission: {sub}")
                    print(evaluate_mod.format_report(m))
            rows.append(row)
    if print_reports:
        print("=" * 70)
    df = pd.DataFrame(rows)
    if not df.empty and evaluate and ctx.assets.has_gt:
        out = ctx.cfg.run_dir / "compare" / "evaluate_table.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def list_outputs(
    ctx: RunContext,
    *,
    options: Sequence[str] = ("motion_only", "two_cam", "ekf"),
    evaluate: bool = True,
) -> pd.DataFrame:
    """Final deliverables. Official metrics table only when ``evaluate`` and GT (eval mode)."""
    if not evaluate or not ctx.assets.has_gt:
        rows: List[dict] = []
        for window in ctx.windows:
            wname = window["name"]
            res = ctx.results.get(wname, {})
            for opt in options:
                if opt not in res:
                    continue
                sub = res[opt]["out_dir"] / "submission.csv"
                rows.append(
                    {
                        "window": wname,
                        "option": opt,
                        "submission_csv": str(sub),
                        "n_rows": len(res[opt]["pred"]),
                    }
                )
                print(f"[{wname} / {opt}] submission: {sub}  rows={len(res[opt]['pred'])}")
        return pd.DataFrame(rows)

    return build_evaluate_table(
        ctx, options=options, evaluate=True, print_reports=True, per_lap=False
    )


# ---------------------------------------------------------------------------
# full run
# ---------------------------------------------------------------------------
def run_all(
    cfg: SubmissionConfig,
    options: Sequence[str] = ("motion_only", "two_cam", "ekf"),
    *,
    show_plots: bool = False,
) -> RunContext:
    ctx = ensure_assets(cfg)
    runner = {"motion_only": run_motion_only, "two_cam": run_two_cam, "ekf": run_ekf}
    for window in ctx.windows:
        print(f"\n===== window {window['name']} =====")
        for opt in options:
            runner[opt](ctx, window, show_plots=show_plots)
    run_summary(ctx, options=options, show_plots=show_plots)
    list_outputs(ctx, options=options, evaluate=ctx.assets.has_gt)
    return ctx
