"""Official-style camloc metrics (eval only; from sfm_hfnet.compute_camloc_metrics).

Option A: map x/y (+ car-frame lat/lon) and yaw.
Option B: racing-line (s, n) via tools/track_frame.py + track-model/map.mcap.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
DEFAULT_MAP_MCAP = HERE / "track-model" / "map.mcap"


def _rms(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v**2))) if len(v) else float("nan")


def _wrap_deg(a: np.ndarray) -> np.ndarray:
    return np.degrees(np.angle(np.exp(1j * np.asarray(a, dtype=np.float64))))


def load_racing_line(map_mcap: Path = DEFAULT_MAP_MCAP):
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    from track_frame import load_lines  # type: ignore

    return load_lines(str(map_mcap))


def compute_camloc_metrics(
    *,
    stamps_ns: np.ndarray,
    gt_xy: np.ndarray,
    gt_yaw: np.ndarray,
    pred_xy: np.ndarray,
    pred_yaw: np.ndarray,
    ok: np.ndarray,
    localize_times_s: Optional[np.ndarray] = None,
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

    if localize_times_s is not None and len(localize_times_s):
        t = np.asarray(localize_times_s, dtype=np.float64)
        out["runtime_ms_mean"] = float(np.mean(t) * 1000)
        out["runtime_ms_median"] = float(np.median(t) * 1000)
        out["runtime_ms_p95"] = float(np.percentile(t, 95) * 1000)
    else:
        out["runtime_ms_mean"] = out["runtime_ms_median"] = out["runtime_ms_p95"] = float(
            "nan"
        )

    if n_ok == 0:
        for k in (
            "pos_rmse_xy_m",
            "lat_rmse_m",
            "lon_rmse_m",
            "yaw_rmse_deg",
            "within_1m_%",
            "within_1m_of_all_%",
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
    yaw_err = _wrap_deg(pyaw - gyaw)

    out["pos_rmse_xy_m"] = _rms(pos)  # Option A score
    out["lat_rmse_m"] = _rms(lat)
    out["lon_rmse_m"] = _rms(lon)
    out["yaw_rmse_deg"] = _rms(yaw_err)
    out["within_1m_%"] = 100.0 * float(np.mean(pos < 1.0))
    out["within_1m_of_all_%"] = 100.0 * float(np.sum(pos < 1.0) / max(n_all, 1))
    out["pos_median_m"] = float(np.median(pos))
    out["pos_p95_m"] = float(np.percentile(pos, 95))

    out["pos_rmse_sn_m"] = float("nan")
    out["s_rmse_m"] = float("nan")
    out["n_rmse_m"] = float("nan")
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
    else:
        out["option_b_error"] = f"map.mcap missing: {map_mcap}"

    return out
