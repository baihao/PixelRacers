"""SE(2) pose fusion + confidence weights (from sfm_hfnet.py / query_demo_ekf.py).

Used by Option 2 (offline two-camera se2 fuse) and by the EKF's per-bucket fuse.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


def pose_confidence(
    ok: bool,
    n_inl: int,
    *,
    e_reproj: Optional[float] = None,
    sigma_reproj: float = 4.0,
) -> float:
    """omega = 1[ok] * n_inl * exp(-e_reproj / sigma)."""
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
    poses: List[Optional[Tuple[float, float, float]]],
    weights: List[float],
    *,
    gate_xy_m: float = 5.0,
    gate_yaw_deg: float = 15.0,
) -> Tuple[Optional[Tuple[float, float, float]], Dict]:
    """omega-weighted SE(2) fuse with disagreement gating."""
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


def fuse_hard_select(
    poses: List[Optional[Tuple[float, float, float]]],
    weights: List[float],
) -> Tuple[Optional[Tuple[float, float, float]], Dict]:
    best_i, best_w = -1, 0.0
    for i, w in enumerate(weights):
        if w > best_w and poses[i] is not None:
            best_i, best_w = i, w
    if best_i < 0:
        return None, {"mode": None, "n_in": 0}
    return poses[best_i], {"mode": "hard_select", "n_in": 1, "index": best_i}


def fuse_loc_tables_se2(
    tables: Dict[str, pd.DataFrame],
    *,
    ref_camera: str = "fr",
    gate_xy_m: float = 5.0,
    gate_yaw_deg: float = 15.0,
) -> pd.DataFrame:
    """Offline nearest-stamp se2 fuse across camera error tables (Option 2).

    Each table needs: stamp_ns, ok, pred_x, pred_y, pred_yaw, n_pnp_inl and
    (optional) gt_x/gt_y/gt_yaw, e_reproj. GT columns pass through from ref.
    """
    cams = list(tables.keys())
    if ref_camera not in tables:
        ref_camera = cams[0]
    ref = tables[ref_camera].sort_values("stamp_ns").reset_index(drop=True)
    aligned = {ref_camera: ref}
    for c in cams:
        if c == ref_camera:
            continue
        df = tables[c].sort_values("stamp_ns").reset_index(drop=True)
        ji = align_by_nearest_stamp(ref["stamp_ns"].to_numpy(), df["stamp_ns"].to_numpy())
        aligned[c] = df.iloc[ji].reset_index(drop=True)

    has_gt = "gt_x" in ref.columns
    rows = []
    for i in range(len(ref)):
        poses: List[Optional[Tuple[float, float, float]]] = []
        weights: List[float] = []
        src_ok = {}
        for c in cams:
            row = aligned[c].iloc[i]
            ok = bool(row.ok)
            src_ok[c] = ok
            if ok and pd.notna(row.pred_x):
                poses.append((float(row.pred_x), float(row.pred_y), float(row.pred_yaw)))
                e_rep = (
                    float(row.e_reproj)
                    if "e_reproj" in row.index and pd.notna(row.e_reproj)
                    else None
                )
                weights.append(pose_confidence(ok, int(row.n_pnp_inl), e_reproj=e_rep))
            else:
                poses.append(None)
                weights.append(0.0)
        fused, info = fuse_se2_weighted(
            poses, weights, gate_xy_m=gate_xy_m, gate_yaw_deg=gate_yaw_deg
        )
        ok_f = fused is not None
        if ok_f:
            px, py, pyaw = fused
        else:
            px = py = pyaw = float("nan")
        row_out = {
            "stamp_ns": int(ref.iloc[i].stamp_ns),
            "ok": ok_f,
            "pred_x": px,
            "pred_y": py,
            "pred_yaw": pyaw,
            "fuse_mode": info.get("mode"),
            "n_pnp_inl": int(max(weights)) if weights else 0,
            "union_ok": bool(any(src_ok.values())),
        }
        if has_gt:
            gx = float(ref.iloc[i].gt_x)
            gy = float(ref.iloc[i].gt_y)
            gyaw = float(ref.iloc[i].gt_yaw)
            row_out["gt_x"] = gx
            row_out["gt_y"] = gy
            row_out["gt_yaw"] = gyaw
            row_out["err_xy"] = (
                float(np.hypot(px - gx, py - gy)) if ok_f else float("nan")
            )
        for c in cams:
            row_out[f"ok_{c}"] = src_ok[c]
        rows.append(row_out)
    return pd.DataFrame(rows)
