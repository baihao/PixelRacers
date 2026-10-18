"""SE dead-reckon + SE(2) EKF + camera sync/extrapolation (from query_demo_ekf.py).

Option 1 (motion only): dead_reckon_se sampled at two-camera t_last.
Option 3 (EKF 2cam):    SE predict + per-cam PnP extrapolated to t_last -> se2
                        fuse -> EKF update with R(s)=diag(sigma(s)^2, .., sigma_yaw^2).

Process noise is read from error-model/se_step_residual_noise.json (MAD sigma)
and written to config/process_noise.yaml.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from fusion import fuse_se2_weighted, pose_confidence, align_by_nearest_stamp

HERE = Path(__file__).resolve().parent
NOISE_JSON = HERE / "error-model" / "se_step_residual_noise.json"
PROCESS_YAML = HERE / "config" / "process_noise.yaml"


def wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


# ---------------------------------------------------------------------------
# process noise
# ---------------------------------------------------------------------------
def write_process_noise_yaml(
    noise_json: Path = NOISE_JSON,
    out_yaml: Path = PROCESS_YAML,
) -> dict:
    """Generate process_noise.yaml from se_step_residual_noise.json mad_sigma."""
    import yaml

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
        "gate_xy_m": 5.0,
        "gate_yaw_deg": 15.0,
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return cfg


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


# ---------------------------------------------------------------------------
# dead reckon + interpolation + propagation
# ---------------------------------------------------------------------------
def dead_reckon_se(
    se_df: pd.DataFrame,
    *,
    x0: float,
    y0: float,
    yaw0: float,
    use_accel: bool = True,
) -> pd.DataFrame:
    """Integrate vx,vy,ax,ay,wz from an initial pose (no SE x,y updates)."""
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
        gt_df[["timestamp_ns", "x", "y", "yaw"]], query_ns
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


# ---------------------------------------------------------------------------
# SE(2) EKF
# ---------------------------------------------------------------------------
class SE2EKF:
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
        self, dt, vx, vy, ax, ay, wz, *, use_accel: bool = True
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

        F = np.eye(3)
        F[0, 2] = -s * dx_b - c * dy_b
        F[1, 2] = c * dx_b - s * dy_b

        sig_dx = np.hypot(cfg.sigma_vx * dt, 0.5 * cfg.sigma_ax * dt * dt)
        sig_dy = np.hypot(cfg.sigma_vy * dt, 0.5 * cfg.sigma_ay * dt * dt)
        sig_dyaw = cfg.sigma_wz * dt
        Qb = np.diag([sig_dx**2, sig_dy**2, sig_dyaw**2])
        Rwb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        Q = Rwb @ Qb @ Rwb.T
        self.P = F @ self.P @ F.T + Q

    def predict_to(self, se_df: pd.DataFrame, t1_ns: int, *, use_accel: bool = True) -> None:
        assert self.t_ns is not None
        if t1_ns <= self.t_ns:
            return
        ts = se_df["timestamp_ns"].to_numpy(dtype=np.int64)
        t_cur = self.t_ns
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


# ---------------------------------------------------------------------------
# sync buckets + per-bucket fuse
# ---------------------------------------------------------------------------
def build_sync_tlast_from_stamps(
    stamps: Dict[str, np.ndarray],
    *,
    ref_camera: str,
    cams: Sequence[str],
) -> np.ndarray:
    """t_last timeline from camera stamps only (no localization needed).

    Anchored on ref_camera; for each ref stamp, take the nearest stamp of the
    other cameras and set t_last = max over cameras. Used by Option 1
    (motion-only), which must NOT run any camera features/PnP.
    """
    ref = np.sort(np.asarray(stamps[ref_camera], dtype=np.int64))
    cols = {ref_camera: ref}
    for c in cams:
        if c == ref_camera:
            continue
        other = np.sort(np.asarray(stamps[c], dtype=np.int64))
        idx = align_by_nearest_stamp(ref, other)
        cols[c] = other[idx]
    stacked = np.vstack([cols[c] for c in cams])
    return stacked.max(axis=0).astype(np.int64)


def build_sync_buckets(
    tables: Dict[str, pd.DataFrame],
    *,
    ref_camera: str,
    cams: Sequence[str],
) -> List[dict]:
    """Nearest-stamp sync anchored on ref camera; t_last = max stamps."""
    ref = tables[ref_camera].sort_values("stamp_ns").reset_index(drop=True)
    aligned_idx = {ref_camera: np.arange(len(ref), dtype=np.int64)}
    srt = {ref_camera: ref}
    for c in cams:
        if c == ref_camera:
            continue
        df = tables[c].sort_values("stamp_ns").reset_index(drop=True)
        srt[c] = df
        aligned_idx[c] = align_by_nearest_stamp(
            ref["stamp_ns"].to_numpy(), df["stamp_ns"].to_numpy()
        )
    buckets = []
    for i in range(len(ref)):
        stamps = {}
        rows = {}
        for c in cams:
            j = int(aligned_idx[c][i])
            row = srt[c].iloc[j]
            stamps[c] = int(row.stamp_ns)
            rows[c] = row
        t_last = max(stamps.values())
        buckets.append({"t_last": t_last, "stamps": stamps, "rows": rows, "ref_i": i})
    return buckets


def fuse_bucket_at_tlast(
    bucket: dict,
    se_df: pd.DataFrame,
    *,
    cams: Sequence[str],
    gate_xy_m: float = 5.0,
    gate_yaw_deg: float = 15.0,
) -> Tuple[Optional[Tuple[float, float, float]], dict]:
    """Propagate each cam pose to t_last then fuse_se2_weighted."""
    t_last = int(bucket["t_last"])
    poses: List[Optional[Tuple[float, float, float]]] = []
    weights: List[float] = []
    info_cams = {}
    for c in cams:
        row = bucket["rows"][c]
        ok = bool(row.ok)
        t_c = int(bucket["stamps"][c])
        if ok and pd.notna(row.pred_x):
            x, y, yaw = float(row.pred_x), float(row.pred_y), float(row.pred_yaw)
            x, y, yaw = propagate_pose(x, y, yaw, se_df, t_c, t_last)
            poses.append((x, y, yaw))
            e_rep = (
                float(row.e_reproj)
                if "e_reproj" in row.index and pd.notna(row.e_reproj)
                else None
            )
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


def observation_R(s_m: float, cfg: NoiseCfg, sigma_of_s_fn) -> np.ndarray:
    sig = float(sigma_of_s_fn(s_m))
    sig = max(sig, 0.05)
    return np.diag([sig**2, sig**2, cfg.sigma_yaw_obs**2])


def s_of_pose(x, y, yaw, t_ns, track_line) -> float:
    s, n, epsi, _ = track_line.xy_to_sn(
        np.array([x]), np.array([y]), np.array([yaw]), np.array([t_ns], dtype=np.int64)
    )
    return float(s[0])


# ---------------------------------------------------------------------------
# Option 1 / Option 3 runners
# ---------------------------------------------------------------------------
def run_mode_a_se_predict_at(
    se_df: pd.DataFrame,
    t_last: np.ndarray,
    *,
    x0: float,
    y0: float,
    yaw0: float,
) -> pd.DataFrame:
    """Option 1: pure SE dead-reckon sampled at the given t_last timeline.

    No camera features/PnP — only SE motion integration + sampling.
    """
    dr = dead_reckon_se(se_df, x0=x0, y0=y0, yaw0=yaw0, use_accel=True)
    return interp_pose(dr, np.asarray(t_last, dtype=np.int64))


def run_mode_a_se_predict(
    se_df: pd.DataFrame,
    buckets: List[dict],
    *,
    x0: float,
    y0: float,
    yaw0: float,
) -> pd.DataFrame:
    """Option 1 (bucket form): pure SE dead-reckon sampled at bucket t_last."""
    t_last = np.array([b["t_last"] for b in buckets], dtype=np.int64)
    return run_mode_a_se_predict_at(se_df, t_last, x0=x0, y0=y0, yaw0=yaw0)


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
    cams: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Option 3: SE predict + per-cam extrapolation + se2 fuse + EKF update(R(s))."""
    cams = tuple(cams)
    ekf = SE2EKF(x0, y0, yaw0, cfg)
    ekf.t_ns = int(t0_ns)
    rows = []
    sim_rows = []
    for b in buckets:
        t_last = int(b["t_last"])
        ekf.predict_to(se_df, t_last)
        fused, finfo = fuse_bucket_at_tlast(
            b, se_df, cams=cams, gate_xy_m=cfg.gate_xy_m, gate_yaw_deg=cfg.gate_yaw_deg
        )
        updated = False
        sig_s = float("nan")
        if fused is not None:
            s_hat = s_of_pose(ekf.x, ekf.y, ekf.yaw, t_last, track_line) if track_line is not None else 0.0
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["timestamp_ns", "x", "y", "yaw"]].to_csv(path, index=False)


def write_gt_lap_slice(gt_all: pd.DataFrame, start_ns: int, end_ns: int, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sl = gt_all[
        (gt_all["timestamp_ns"] >= start_ns) & (gt_all["timestamp_ns"] <= end_ns)
    ][["timestamp_ns", "x", "y", "yaw"]].copy()
    sl.to_csv(path, index=False)
    return path
