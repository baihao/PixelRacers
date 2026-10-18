"""End-to-end submission pipeline: assets -> localize (cached) -> 3 options.

Config via SubmissionConfig. Three options map to three cells / CLI subcommands:
  Option 1  motion_only  — SE dead-reckon sampled at two-camera t_last
  Option 2  two_cam      — offline fr+rl se2 fuse (no EKF)
  Option 3  ekf          — SE predict + fr/rl extrapolate -> se2 -> EKF update R(s)

MODE="eval" (training+GT): also writes metrics + cropped GT.
MODE="predict" (test):     submission.csv only.
"""
from __future__ import annotations

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
def ensure_assets(cfg: SubmissionConfig) -> RunContext:
    """Extract mcap assets, load calibs + racing line, resolve windows."""
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
    outline = plots.load_track_outline(assets_dir / "track_bounds.json")

    print(f"[setup] mode={cfg.mode} windows={[w['name'] for w in windows]}")
    print(f"[setup] cams={tuple(cfg.cams)} backend={cfg.backend} has_gt={cfg.has_gt}")
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


def ensure_loc_tables(ctx: RunContext, window: Dict) -> Dict[str, pd.DataFrame]:
    """Online localize (or cache hit) for each camera over the window."""
    cfg = ctx.cfg
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


def _init_pose(ctx: RunContext, se_win: pd.DataFrame) -> Tuple[float, float, float, int]:
    """First-frame init pose: GT if available (eval), else SE first pose."""
    if ctx.assets.has_gt:
        gt = ctx.assets.gt
        t0 = int(se_win.iloc[0].timestamp_ns)
        g = gt.iloc[(gt.timestamp_ns - t0).abs().argmin()]
        return float(g.x), float(g.y), float(g.yaw), t0
    r0 = se_win.iloc[0]
    return float(r0.x), float(r0.y), float(r0.yaw), int(r0.timestamp_ns)


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
    x0, y0, yaw0, _ = _init_pose(ctx, se_win)
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
    x0, y0, yaw0, t0 = _init_pose(ctx, se_win)
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
def run_summary(
    ctx: RunContext,
    *,
    options: Sequence[str] = ("motion_only", "two_cam", "ekf"),
    show_plots: bool = True,
) -> Optional[pd.DataFrame]:
    """Combined comparison: metrics table + metrics bars + trajectory overlay.

    eval: writes compare/metrics_compare.csv and comparison figures.
    predict: lists submission rows and draws trajectory overlay only.
    """
    cfg = ctx.cfg
    cmp_dir = cfg.run_dir / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    mdf_out: Optional[pd.DataFrame] = None

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
    import subprocess

    evaluate_py = TOOLS / "evaluate.py"
    r = subprocess.run(
        [sys.executable, str(evaluate_py), str(gt_csv), str(submission_csv)],
        capture_output=True,
        text=True,
        check=False,
    )
    return (r.stdout or "") + (r.stderr or "")


def list_outputs(
    ctx: RunContext,
    *,
    options: Sequence[str] = ("motion_only", "two_cam", "ekf"),
    evaluate: bool = True,
) -> pd.DataFrame:
    """Final deliverables: submission.csv path per option; eval -> evaluate.py.

    Returns a DataFrame of {window, option, submission_csv, (score columns)} and
    prints each option's submission path plus the official evaluate.py report.
    """
    rows = []
    for window in ctx.windows:
        wname = window["name"]
        res = ctx.results.get(wname, {})
        gt_csv = gt_slice_path(ctx, window) if ctx.assets.has_gt else None
        for opt in options:
            if opt not in res:
                continue
            sub = res[opt]["out_dir"] / "submission.csv"
            row = {"window": wname, "option": opt, "submission_csv": str(sub)}
            print("=" * 70)
            print(f"[{wname} / {opt}] submission: {sub}")
            if ctx.assets.has_gt and gt_csv is not None:
                report = run_evaluate(gt_csv, sub)
                print(report.rstrip())
                row["gt_csv"] = str(gt_csv)
                for line in report.splitlines():
                    if "position RMSE" in line and "SCORE" in line:
                        # ".. position RMSE  [m]  <num>  <- SCORE .."
                        for tok in line.split():
                            try:
                                row["score_pos_rmse_m"] = float(tok)
                                break
                            except ValueError:
                                continue
            rows.append(row)
    print("=" * 70)
    return pd.DataFrame(rows)


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
