"""Lap3 EKF query-demo helpers: SE dead-reckon, 3-cam sync+propagate, EKF update."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CAMLOC = ROOT.parent / "c0_2026-10-13_22-53-42_camloc"
EXTRACTED = CAMLOC / "extracted"
LAPDATA = ROOT / "lapdata"
# CLAHE + highlight-filter maps (fr/rl Lap2+4); see sfm_hfnet_fr_rl_clahe_lap3_loc.ipynb
SFM_OUT = ROOT / "3d-map-test" / "outputs_hfnet_clahe_fr_rl"
DEFAULT_LOC_CAMERAS: Tuple[str, ...] = ("fr", "rl")
NOISE_JSON = ROOT / "state-error-estimation" / "se_step_residual_noise.json"
PROCESS_YAML = HERE / "config" / "process_noise.yaml"
EVALUATE_PY = CAMLOC / "tools" / "evaluate.py"
GT_CSV = EXTRACTED / "gt.csv"
DEFAULT_MAP_MCAP = CAMLOC / "23-00-38_cut_120m_update" / "map.mcap"


def ensure_paths() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "3d-map-test"))
    tools = DEFAULT_MAP_MCAP.parent / "tools"
    if tools.is_dir() and str(tools) not in sys.path:
        sys.path.insert(0, str(tools))


def load_racing_line(map_mcap: Path = DEFAULT_MAP_MCAP):
    ensure_paths()
    from track_frame import load_lines  # type: ignore

    return load_lines(str(map_mcap))


def _rms(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    return float(np.sqrt(np.mean(v**2))) if len(v) else float("nan")


def pose_confidence(
    ok: bool,
    n_inl: int,
    *,
    e_reproj: Optional[float] = None,
    sigma_reproj: float = 4.0,
) -> float:
    if not ok or n_inl <= 0:
        return 0.0
    w = float(n_inl)
    if e_reproj is not None and np.isfinite(e_reproj):
        w *= float(np.exp(-float(e_reproj) / max(sigma_reproj, 1e-6)))
    return w


def align_by_nearest_stamp(ref_stamps: np.ndarray, other_stamps: np.ndarray) -> np.ndarray:
    ref_stamps = np.asarray(ref_stamps, dtype=np.int64)
    other_stamps = np.asarray(other_stamps, dtype=np.int64)
    idx = np.searchsorted(other_stamps, ref_stamps)
    out = np.empty(len(ref_stamps), dtype=np.int64)
    for i, (t, j) in enumerate(zip(ref_stamps, idx)):
        cands = [k for k in (int(j), int(j) - 1) if 0 <= k < len(other_stamps)]
        out[i] = min(cands, key=lambda k: abs(int(other_stamps[k]) - int(t)))
    return out


def fuse_se2_weighted(
    poses: List[Tuple[float, float, float]],
    weights: List[float],
    *,
    gate_xy_m: float = 5.0,
    gate_yaw_deg: float = 15.0,
) -> Tuple[Optional[Tuple[float, float, float]], Dict]:
    items = [(p, float(w)) for p, w in zip(poses, weights) if w > 0 and p is not None]
    info: Dict = {"n_in": len(items), "mode": None, "weights": []}
    if not items:
        return None, info
    items.sort(key=lambda pw: -pw[1])
    (x0, y0, yaw0), w0 = items[0]
    kept = [(x0, y0, yaw0, w0)]
    for (x, y, yaw), w in items[1:]:
        dxy = float(np.hypot(x - x0, y - y0))
        dyaw = abs(float(wrap_pi(yaw - yaw0))) * 180 / np.pi
        if dxy <= gate_xy_m and dyaw <= gate_yaw_deg:
            kept.append((x, y, yaw, w))
    info["n_kept"] = len(kept)
    info["weights"] = [k[3] for k in kept]
    if len(kept) == 1:
        info["mode"] = "single"
        return (kept[0][0], kept[0][1], kept[0][2]), info
    ws = np.array([k[3] for k in kept], dtype=np.float64)
    ws = ws / ws.sum()
    x = float(np.sum(ws * np.array([k[0] for k in kept])))
    y = float(np.sum(ws * np.array([k[1] for k in kept])))
    sy = float(np.sum(ws * np.sin([k[2] for k in kept])))
    cy = float(np.sum(ws * np.cos([k[2] for k in kept])))
    yaw = float(np.arctan2(sy, cy))
    info["mode"] = "se2_weighted"
    return (x, y, yaw), info


def compute_camloc_metrics(
    *,
    stamps_ns: np.ndarray,
    gt_xy: np.ndarray,
    gt_yaw: np.ndarray,
    pred_xy: np.ndarray,
    pred_yaw: np.ndarray,
    ok: np.ndarray,
    map_mcap: Path = DEFAULT_MAP_MCAP,
) -> Dict:
    ok = np.asarray(ok, dtype=bool)
    n_all = int(len(ok))
    n_ok = int(ok.sum())
    out: Dict = {
        "n_query": n_all,
        "n_success": n_ok,
        "success_rate_%": 100.0 * n_ok / max(n_all, 1),
    }
    if n_ok == 0:
        for k in (
            "pos_rmse_xy_m",
            "lat_rmse_m",
            "lon_rmse_m",
            "yaw_rmse_deg",
            "within_1m_%",
            "pos_rmse_sn_m",
            "s_rmse_m",
            "n_rmse_m",
        ):
            out[k] = float("nan")
        return out
    gx, gy = gt_xy[ok, 0], gt_xy[ok, 1]
    px, py = pred_xy[ok, 0], pred_xy[ok, 1]
    gyaw, pyaw = gt_yaw[ok], pred_yaw[ok]
    dx, dy = px - gx, py - gy
    pos = np.hypot(dx, dy)
    lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
    lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
    yaw_err = np.degrees(wrap_pi(pyaw - gyaw))
    out["pos_rmse_xy_m"] = _rms(pos)
    out["lat_rmse_m"] = _rms(lat)
    out["lon_rmse_m"] = _rms(lon)
    out["yaw_rmse_deg"] = _rms(yaw_err)
    # within 1 m: denominator = all queries (failed frames do not count as within 1 m)
    out["within_1m_%"] = 100.0 * float(np.sum(pos < 1.0) / max(n_all, 1))
    out["pos_median_m"] = float(np.median(pos))
    out["pos_p95_m"] = float(np.percentile(pos, 95))
    out["pos_rmse_sn_m"] = out["s_rmse_m"] = out["n_rmse_m"] = float("nan")
    map_mcap = Path(map_mcap)
    if map_mcap.is_file():
        lines = load_racing_line(map_mcap)
        lap = lines["track"].lap
        ts = np.asarray(stamps_ns, dtype=np.int64)[ok]
        sg, ng, _, _ = lines["track"].xy_to_sn(gx, gy, gyaw, ts)
        sp, np_, _, _ = lines["track"].xy_to_sn(px, py, pyaw, ts)
        sg_u = np.unwrap(sg, period=lap)
        ds = (sp - sg_u + lap / 2) % lap - lap / 2
        dn = np_ - ng
        out["s_rmse_m"] = _rms(ds)
        out["n_rmse_m"] = _rms(dn)
        out["pos_rmse_sn_m"] = _rms(np.hypot(ds, dn))
        out["track_lap_m"] = float(lap)
    return out


def read_csv_ns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"timestamp_ns": str})
    df["timestamp_ns"] = df["timestamp_ns"].astype("int64")
    return df


def load_lap_window(lap: int = 3) -> Tuple[int, int, dict]:
    with open(LAPDATA / "laps.json") as f:
        meta = json.load(f)
    row = next(L for L in meta["laps"] if L["lap"] == lap)
    return int(row["start_ns"]), int(row["end_ns"]), row


def write_process_noise_yaml(
    noise_json: Path = NOISE_JSON,
    out_yaml: Path = PROCESS_YAML,
) -> dict:
    """Generate process_noise.yaml from se_step_residual_noise.json mad_sigma."""
    model = json.loads(Path(noise_json).read_text())
    vx, vy, _ = model["velocity_body"]["mad_sigma"]
    ax, ay, _ = model["acceleration_body"]["mad_sigma"]
    wz = float(model["wz_minus_dyaw"]["mad_sigma"][0])
    cfg = {
        "source": str(noise_json),
        "lap_filter": model.get("lap_filter"),
        "sigma_vx_mps": float(vx),
        "sigma_vy_mps": float(vy),
        "sigma_ax_mps2": float(ax),
        "sigma_ay_mps2": float(ay),
        "sigma_wz_radps": wz,
        "P0_xy_m2": 0.25,
        "P0_yaw_rad2": float(np.deg2rad(1.0) ** 2),
        "sigma_yaw_obs_rad": float(np.deg2rad(2.0)),
        "tau_sync_ms": 50,
        "gate_xy_m": 5.0,
        "gate_yaw_deg": 15.0,
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return cfg


def load_process_noise(path: Path = PROCESS_YAML) -> dict:
    if not path.is_file():
        return write_process_noise_yaml(out_yaml=path)
    return yaml.safe_load(path.read_text())


def wrap_pi(a: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(a), np.cos(a))


def dead_reckon_se(
    se_df: pd.DataFrame,
    *,
    x0: float,
    y0: float,
    yaw0: float,
    use_accel: bool = True,
) -> pd.DataFrame:
    """Integrate vx,vy,ax,ay,wz from first SE pose (no SE x,y updates)."""
    ts = se_df["timestamp_ns"].to_numpy(dtype=np.int64)
    vx = se_df["vx_mps"].to_numpy(dtype=np.float64)
    vy = se_df["vy_mps"].to_numpy(dtype=np.float64)
    ax = se_df["ax_mps2"].to_numpy(dtype=np.float64)
    ay = se_df["ay_mps2"].to_numpy(dtype=np.float64)
    wz = se_df["wz_radps"].to_numpy(dtype=np.float64)

    n = len(se_df)
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    yaws = np.empty(n, dtype=np.float64)
    xs[0], ys[0], yaws[0] = x0, y0, yaw0
    x, y, yaw = x0, y0, yaw0
    for i in range(n - 1):
        dt = (ts[i + 1] - ts[i]) * 1e-9
        if dt <= 0:
            xs[i + 1], ys[i + 1], yaws[i + 1] = x, y, yaw
            continue
        dx_b = vx[i] * dt
        dy_b = vy[i] * dt
        if use_accel:
            dx_b += 0.5 * ax[i] * dt * dt
            dy_b += 0.5 * ay[i] * dt * dt
        yaw_mid = yaw + 0.5 * wz[i] * dt
        c, s = np.cos(yaw_mid), np.sin(yaw_mid)
        x = x + c * dx_b - s * dy_b
        y = y + s * dx_b + c * dy_b
        yaw = float(wrap_pi(yaw + wz[i] * dt))
        xs[i + 1], ys[i + 1], yaws[i + 1] = x, y, yaw
    return pd.DataFrame({"timestamp_ns": ts, "x": xs, "y": ys, "yaw": yaws})


def interp_pose(pose_df: pd.DataFrame, query_ns: np.ndarray) -> pd.DataFrame:
    t = pose_df["timestamp_ns"].to_numpy(dtype=np.float64)
    x = pose_df["x"].to_numpy(dtype=np.float64)
    y = pose_df["y"].to_numpy(dtype=np.float64)
    yaw = np.unwrap(pose_df["yaw"].to_numpy(dtype=np.float64))
    q = query_ns.astype(np.float64)
    return pd.DataFrame(
        {
            "timestamp_ns": query_ns.astype(np.int64),
            "x": np.interp(q, t, x),
            "y": np.interp(q, t, y),
            "yaw": wrap_pi(np.interp(q, t, yaw)),
        }
    )


def interp_gt(gt_df: pd.DataFrame, query_ns: np.ndarray) -> pd.DataFrame:
    return interp_pose(
        gt_df.rename(columns={"timestamp_ns": "timestamp_ns"})[
            ["timestamp_ns", "x", "y", "yaw"]
        ],
        query_ns,
    )


def se_control_at(se_df: pd.DataFrame, t_ns: int) -> Tuple[float, float, float, float, float]:
    """Nearest SE control u = (vx,vy,ax,ay,wz) at t_ns."""
    ts = se_df["timestamp_ns"].to_numpy(dtype=np.int64)
    j = int(np.searchsorted(ts, t_ns))
    j = min(max(j, 0), len(ts) - 1)
    if j > 0 and abs(int(ts[j - 1]) - t_ns) < abs(int(ts[j]) - t_ns):
        j = j - 1
    row = se_df.iloc[j]
    return (
        float(row.vx_mps),
        float(row.vy_mps),
        float(row.ax_mps2),
        float(row.ay_mps2),
        float(row.wz_radps),
    )


def propagate_pose(
    x: float,
    y: float,
    yaw: float,
    se_df: pd.DataFrame,
    t0_ns: int,
    t1_ns: int,
    *,
    use_accel: bool = True,
) -> Tuple[float, float, float]:
    """Propagate pose from t0 to t1 using SE v/a/wz (same model as dead_reckon)."""
    if t1_ns <= t0_ns:
        return x, y, yaw
    ts = se_df["timestamp_ns"].to_numpy(dtype=np.int64)
    # control index: last SE sample at or before current time
    j = int(np.searchsorted(ts, t0_ns, side="right")) - 1
    j = max(0, min(j, len(ts) - 1))
    t_cur = int(t0_ns)
    while t_cur < t1_ns:
        t_next_se = int(ts[j + 1]) if j + 1 < len(ts) else int(t1_ns)
        t_next = min(t_next_se, int(t1_ns))
        dt = (t_next - t_cur) * 1e-9
        if dt > 0:
            row = se_df.iloc[j]
            vx, vy = float(row.vx_mps), float(row.vy_mps)
            ax, ay = float(row.ax_mps2), float(row.ay_mps2)
            wz = float(row.wz_radps)
            dx_b = vx * dt
            dy_b = vy * dt
            if use_accel:
                dx_b += 0.5 * ax * dt * dt
                dy_b += 0.5 * ay * dt * dt
            yaw_mid = yaw + 0.5 * wz * dt
            c, s = np.cos(yaw_mid), np.sin(yaw_mid)
            x = x + c * dx_b - s * dy_b
            y = y + s * dx_b + c * dy_b
            yaw = float(wrap_pi(yaw + wz * dt))
        t_cur = t_next
        if t_cur >= t1_ns:
            break
        j += 1
        if j >= len(ts):
            break
    return x, y, yaw


@dataclass
class NoiseCfg:
    sigma_vx: float
    sigma_vy: float
    sigma_ax: float
    sigma_ay: float
    sigma_wz: float
    P0_xy: float
    P0_yaw: float
    sigma_yaw_obs: float
    gate_xy_m: float
    gate_yaw_deg: float

    @classmethod
    def from_dict(cls, d: dict) -> "NoiseCfg":
        return cls(
            sigma_vx=float(d["sigma_vx_mps"]),
            sigma_vy=float(d["sigma_vy_mps"]),
            sigma_ax=float(d["sigma_ax_mps2"]),
            sigma_ay=float(d["sigma_ay_mps2"]),
            sigma_wz=float(d["sigma_wz_radps"]),
            P0_xy=float(d["P0_xy_m2"]),
            P0_yaw=float(d["P0_yaw_rad2"]),
            sigma_yaw_obs=float(d["sigma_yaw_obs_rad"]),
            gate_xy_m=float(d.get("gate_xy_m", 5.0)),
            gate_yaw_deg=float(d.get("gate_yaw_deg", 15.0)),
        )


class SE2EKF:
    """Minimal SE(2) EKF with body-velocity control."""

    def __init__(self, x0: float, y0: float, yaw0: float, cfg: NoiseCfg):
        self.x = float(x0)
        self.y = float(y0)
        self.yaw = float(yaw0)
        self.P = np.diag([cfg.P0_xy, cfg.P0_xy, cfg.P0_yaw]).astype(np.float64)
        self.cfg = cfg
        self.t_ns: Optional[int] = None

    def pose(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.yaw

    def predict_step(
        self,
        dt: float,
        vx: float,
        vy: float,
        ax: float,
        ay: float,
        wz: float,
        *,
        use_accel: bool = True,
    ) -> None:
        if dt <= 0:
            return
        cfg = self.cfg
        dx_b = vx * dt
        dy_b = vy * dt
        if use_accel:
            dx_b += 0.5 * ax * dt * dt
            dy_b += 0.5 * ay * dt * dt
        yaw_mid = self.yaw + 0.5 * wz * dt
        c, s = np.cos(yaw_mid), np.sin(yaw_mid)
        self.x = self.x + c * dx_b - s * dy_b
        self.y = self.y + s * dx_b + c * dy_b
        self.yaw = float(wrap_pi(self.yaw + wz * dt))

        # Jacobian F (approx, mid-yaw)
        F = np.eye(3)
        F[0, 2] = -s * dx_b - c * dy_b
        F[1, 2] = c * dx_b - s * dy_b

        # Control noise → body displacement / yaw, then rotate xy
        sig_dx = np.hypot(cfg.sigma_vx * dt, 0.5 * cfg.sigma_ax * dt * dt)
        sig_dy = np.hypot(cfg.sigma_vy * dt, 0.5 * cfg.sigma_ay * dt * dt)
        sig_dyaw = cfg.sigma_wz * dt
        Qb = np.diag([sig_dx**2, sig_dy**2, sig_dyaw**2])
        Rwb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        Q = Rwb @ Qb @ Rwb.T
        self.P = F @ self.P @ F.T + Q

    def predict_to(
        self,
        se_df: pd.DataFrame,
        t1_ns: int,
        *,
        use_accel: bool = True,
    ) -> None:
        assert self.t_ns is not None
        if t1_ns <= self.t_ns:
            return
        ts = se_df["timestamp_ns"].to_numpy(dtype=np.int64)
        i0 = int(np.searchsorted(ts, self.t_ns, side="left"))
        # walk SE samples
        t_cur = self.t_ns
        # index of last SE <= t_cur
        j = int(np.searchsorted(ts, t_cur, side="right")) - 1
        j = max(0, j)
        while True:
            t_next_se = int(ts[j + 1]) if j + 1 < len(ts) else t1_ns
            t_next = min(t_next_se, t1_ns)
            dt = (t_next - t_cur) * 1e-9
            row = se_df.iloc[j]
            self.predict_step(
                dt,
                float(row.vx_mps),
                float(row.vy_mps),
                float(row.ax_mps2),
                float(row.ay_mps2),
                float(row.wz_radps),
                use_accel=use_accel,
            )
            t_cur = t_next
            if t_cur >= t1_ns:
                break
            j += 1
            if j >= len(ts):
                break
        self.t_ns = t1_ns

    def update(self, zx: float, zy: float, zyaw: float, R: np.ndarray) -> None:
        H = np.eye(3)
        z = np.array([zx, zy, zyaw], dtype=np.float64)
        h = np.array([self.x, self.y, self.yaw], dtype=np.float64)
        y = z - h
        y[2] = float(wrap_pi(y[2]))
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        self.x += float(dx[0])
        self.y += float(dx[1])
        self.yaw = float(wrap_pi(self.yaw + dx[2]))
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P


def load_cam_loc_tables(
    cams: Sequence[str] = ("r", "fr", "rl"),
    loc_root: Path = SFM_OUT,
    *,
    query_lap: int = 3,
) -> Dict[str, pd.DataFrame]:
    tables = {}
    for c in cams:
        path = loc_root / f"loc_lap{int(query_lap)}_{c}" / "errors.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run SfM query for lap {query_lap} first "
                f"(map_*_lap2+4 → loc_lap{query_lap}_{{cam}})"
            )
        df = pd.read_csv(path)
        df["stamp_ns"] = df["stamp_ns"].astype("int64")
        tables[c] = df.sort_values("stamp_ns").reset_index(drop=True)
    return tables


def fuse_loc_tables_se2(
    tables: Dict[str, pd.DataFrame],
    *,
    ref_camera: str = "r",
    gate_xy_m: float = 5.0,
    gate_yaw_deg: float = 15.0,
) -> pd.DataFrame:
    """Offline nearest-stamp se2 fuse (no EKF / no time propagate)."""
    cams = list(tables.keys())
    if ref_camera not in tables:
        ref_camera = cams[0]
    ref = tables[ref_camera].sort_values("stamp_ns").reset_index(drop=True)
    aligned = {ref_camera: ref}
    for c in cams:
        if c == ref_camera:
            continue
        df = tables[c].sort_values("stamp_ns").reset_index(drop=True)
        ji = align_by_nearest_stamp(
            ref["stamp_ns"].to_numpy(), df["stamp_ns"].to_numpy()
        )
        aligned[c] = df.iloc[ji].reset_index(drop=True)

    rows = []
    for i in range(len(ref)):
        poses, weights = [], []
        src_ok = {}
        for c in cams:
            row = aligned[c].iloc[i]
            ok = bool(row.ok)
            src_ok[c] = ok
            if ok and pd.notna(row.pred_x):
                poses.append(
                    (float(row.pred_x), float(row.pred_y), float(row.pred_yaw))
                )
                e_rep = (
                    float(row.e_reproj)
                    if "e_reproj" in row.index and pd.notna(row.e_reproj)
                    else None
                )
                weights.append(
                    pose_confidence(ok, int(row.n_pnp_inl), e_reproj=e_rep)
                )
            else:
                poses.append(None)
                weights.append(0.0)
        fused, info = fuse_se2_weighted(
            poses, weights, gate_xy_m=gate_xy_m, gate_yaw_deg=gate_yaw_deg
        )
        gx = float(ref.iloc[i].gt_x)
        gy = float(ref.iloc[i].gt_y)
        gyaw = float(ref.iloc[i].gt_yaw)
        ok_f = fused is not None
        if ok_f:
            px, py, pyaw = fused
            err_xy = float(np.hypot(px - gx, py - gy))
        else:
            px = py = pyaw = err_xy = float("nan")
        row_out = {
            "stamp_ns": int(ref.iloc[i].stamp_ns),
            "ok": ok_f,
            "gt_x": gx,
            "gt_y": gy,
            "gt_yaw": gyaw,
            "pred_x": px,
            "pred_y": py,
            "pred_yaw": pyaw,
            "err_xy": err_xy,
            "fuse_mode": info.get("mode"),
            "n_pnp_inl": int(max(weights)) if weights else 0,
            "union_ok": bool(any(src_ok.values())),
        }
        for c in cams:
            row_out[f"ok_{c}"] = src_ok[c]
        rows.append(row_out)
    return pd.DataFrame(rows)


def build_sync_buckets(
    tables: Dict[str, pd.DataFrame],
    *,
    ref_camera: str = "r",
    cams: Sequence[str] = ("r", "fr", "rl"),
) -> List[dict]:
    """Nearest-stamp sync anchored on ref camera; t_last = max stamps."""
    ref = tables[ref_camera]
    aligned_idx = {ref_camera: np.arange(len(ref), dtype=np.int64)}
    for c in cams:
        if c == ref_camera:
            continue
        aligned_idx[c] = align_by_nearest_stamp(
            ref["stamp_ns"].to_numpy(), tables[c]["stamp_ns"].to_numpy()
        )

    buckets = []
    for i in range(len(ref)):
        stamps = {}
        rows = {}
        for c in cams:
            j = int(aligned_idx[c][i])
            row = tables[c].iloc[j]
            stamps[c] = int(row.stamp_ns)
            rows[c] = row
        t_last = max(stamps.values())
        buckets.append({"t_last": t_last, "stamps": stamps, "rows": rows, "ref_i": i})
    return buckets


def fuse_bucket_at_tlast(
    bucket: dict,
    se_df: pd.DataFrame,
    *,
    cams: Sequence[str] = ("r", "fr", "rl"),
    gate_xy_m: float = 5.0,
    gate_yaw_deg: float = 15.0,
) -> Tuple[Optional[Tuple[float, float, float]], dict]:
    """Propagate each cam pose to t_last then fuse_se2_weighted."""
    t_last = int(bucket["t_last"])
    poses = []
    weights = []
    info_cams = {}
    for c in cams:
        row = bucket["rows"][c]
        ok = bool(row.ok)
        t_c = int(bucket["stamps"][c])
        if ok and pd.notna(row.pred_x):
            x, y, yaw = float(row.pred_x), float(row.pred_y), float(row.pred_yaw)
            x, y, yaw = propagate_pose(x, y, yaw, se_df, t_c, t_last)
            poses.append((x, y, yaw))
            e_rep = float(row.e_reproj) if "e_reproj" in row.index and pd.notna(row.e_reproj) else None
            w = pose_confidence(ok, int(row.n_pnp_inl), e_reproj=e_rep)
            weights.append(w)
            info_cams[c] = {"ok": True, "w": w, "dt_ms": (t_last - t_c) * 1e-6}
        else:
            poses.append(None)
            weights.append(0.0)
            info_cams[c] = {"ok": False, "w": 0.0, "dt_ms": (t_last - t_c) * 1e-6}
    fused, finfo = fuse_se2_weighted(
        poses, weights, gate_xy_m=gate_xy_m, gate_yaw_deg=gate_yaw_deg
    )
    finfo["cams"] = info_cams
    return fused, finfo


def observation_R(
    s_m: float,
    cfg: NoiseCfg,
    sigma_of_s_fn,
) -> np.ndarray:
    sig = float(sigma_of_s_fn(s_m))
    sig = max(sig, 0.05)
    return np.diag([sig**2, sig**2, cfg.sigma_yaw_obs**2])


def s_of_pose(x: float, y: float, yaw: float, t_ns: int, track_line) -> float:
    s, n, epsi, _ = track_line.xy_to_sn(
        np.array([x]), np.array([y]), np.array([yaw]), np.array([t_ns], dtype=np.int64)
    )
    return float(s[0])


def run_mode_a_se_predict(
    se_df: pd.DataFrame,
    buckets: List[dict],
    *,
    x0: float,
    y0: float,
    yaw0: float,
) -> pd.DataFrame:
    dr = dead_reckon_se(se_df, x0=x0, y0=y0, yaw0=yaw0, use_accel=True)
    t_last = np.array([b["t_last"] for b in buckets], dtype=np.int64)
    pred = interp_pose(dr, t_last)
    return pred.rename(columns={"timestamp_ns": "timestamp_ns"})


def run_mode_b_ekf(
    se_df: pd.DataFrame,
    buckets: List[dict],
    cfg: NoiseCfg,
    *,
    x0: float,
    y0: float,
    yaw0: float,
    t0_ns: int,
    sigma_of_s_fn,
    track_line=None,
    cams: Sequence[str] = ("r", "fr", "rl"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cams = tuple(cams)
    ekf = SE2EKF(x0, y0, yaw0, cfg)
    ekf.t_ns = int(t0_ns)
    rows = []
    sim_rows = []
    for b in buckets:
        t_last = int(b["t_last"])
        ekf.predict_to(se_df, t_last)
        fused, finfo = fuse_bucket_at_tlast(
            b,
            se_df,
            cams=cams,
            gate_xy_m=cfg.gate_xy_m,
            gate_yaw_deg=cfg.gate_yaw_deg,
        )
        updated = False
        sig_s = float("nan")
        if fused is not None:
            if track_line is not None:
                s_hat = s_of_pose(ekf.x, ekf.y, ekf.yaw, t_last, track_line)
            else:
                s_hat = 0.0
            R = observation_R(s_hat, cfg, sigma_of_s_fn)
            sig_s = float(np.sqrt(R[0, 0]))
            ekf.update(fused[0], fused[1], fused[2], R)
            updated = True
        x, y, yaw = ekf.pose()
        rows.append({"timestamp_ns": t_last, "x": x, "y": y, "yaw": yaw})
        sim_rows.append(
            {
                "timestamp_ns": t_last,
                "x": x,
                "y": y,
                "yaw": yaw,
                "updated": updated,
                "fuse_mode": finfo.get("mode"),
                "n_kept": finfo.get("n_kept", 0),
                "sigma_obs_m": sig_s,
                "Pxx": float(ekf.P[0, 0]),
                "Pyy": float(ekf.P[1, 1]),
                "Pyaw": float(ekf.P[2, 2]),
                **{f"ok_{c}": finfo.get("cams", {}).get(c, {}).get("ok", False) for c in cams},
                **{f"w_{c}": finfo.get("cams", {}).get(c, {}).get("w", 0.0) for c in cams},
                **{f"dt_ms_{c}": finfo.get("cams", {}).get(c, {}).get("dt_ms", 0.0) for c in cams},
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(sim_rows)


def write_submission(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["timestamp_ns", "x", "y", "yaw"]].copy()
    out.to_csv(path, index=False)


def write_gt_lap_slice(
    gt_all: pd.DataFrame,
    start_ns: int,
    end_ns: int,
    path: Path,
) -> Path:
    """Crop GT to lap window so evaluate.py coverage is lap-relative."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sl = gt_all[
        (gt_all["timestamp_ns"] >= start_ns) & (gt_all["timestamp_ns"] <= end_ns)
    ][["timestamp_ns", "x", "y", "yaw"]].copy()
    sl.to_csv(path, index=False)
    return path


def run_evaluate(submission: Path, gt_path: Path = GT_CSV) -> str:
    cmd = [sys.executable, str(EVALUATE_PY), str(gt_path), str(submission)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return (r.stdout or "") + (r.stderr or "")


def metrics_vs_gt(
    pred: pd.DataFrame,
    gt_at: pd.DataFrame,
    *,
    ok: Optional[np.ndarray] = None,
) -> dict:
    if ok is None:
        ok = np.ones(len(pred), dtype=bool)
    return compute_camloc_metrics(
        stamps_ns=pred["timestamp_ns"].to_numpy(),
        gt_xy=np.column_stack([gt_at["x"], gt_at["y"]]),
        gt_yaw=gt_at["yaw"].to_numpy(),
        pred_xy=np.column_stack([pred["x"], pred["y"]]),
        pred_yaw=pred["yaw"].to_numpy(),
        ok=ok,
    )


def load_track_bounds() -> Tuple[np.ndarray, np.ndarray]:
    tb = EXTRACTED / "track_bounds.json"
    with open(tb) as f:
        data = json.load(f)

    def polyline_xy(key: str) -> np.ndarray:
        pts = data[key]["points"]
        return np.array([[p["x"], p["y"]] for p in pts], dtype=np.float64)

    return polyline_xy("track_left"), polyline_xy("track_right")
