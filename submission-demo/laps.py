"""Lap window derivation.

eval:    windows from GT via start-finish plane crossing (same method noted in
         lapdata/laps.json), or from an external laps.json if provided.
predict: no GT -> whole-mcap single window, or user-supplied WINDOWS, or an
         external laps.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def detect_laps_from_gt(
    gt: pd.DataFrame,
    *,
    across_max_m: float = 15.0,
    dist_max_m: float = 40.0,
    debounce_s: float = 30.0,
) -> List[Dict]:
    """Start-finish plane crossings along the heading at the first GT pose.

    Returns a list of lap dicts: {lap, start_ns, end_ns, duration_s} for each
    complete lap (between consecutive crossings).
    """
    g = gt.sort_values("timestamp_ns").reset_index(drop=True)
    ts = g["timestamp_ns"].to_numpy(np.int64)
    x = g["x"].to_numpy(float)
    y = g["y"].to_numpy(float)
    yaw = g["yaw"].to_numpy(float)
    if len(g) < 2:
        return []

    x0, y0, yaw0 = x[0], y[0], yaw[0]
    # start-finish plane: normal = heading direction at first pose
    nx, ny = np.cos(yaw0), np.sin(yaw0)       # "along" (signed progress through plane)
    ax, ay = -np.sin(yaw0), np.cos(yaw0)      # "across" (lateral offset in plane)

    dx, dy = x - x0, y - y0
    along = dx * nx + dy * ny
    across = dx * ax + dy * ay
    dist = np.hypot(dx, dy)

    crossings: List[int] = []
    last_cross_ns: Optional[int] = None
    for i in range(1, len(g)):
        # crossing the plane along-direction (sign change of `along`) near origin
        if along[i - 1] < 0.0 <= along[i]:
            if abs(across[i]) < across_max_m and dist[i] < dist_max_m:
                if last_cross_ns is None or (ts[i] - last_cross_ns) * 1e-9 > debounce_s:
                    crossings.append(i)
                    last_cross_ns = ts[i]

    laps: List[Dict] = []
    for k in range(len(crossings) - 1):
        i0 = crossings[k]
        i1 = crossings[k + 1]
        s_ns = int(ts[i0])
        e_ns = int(ts[i1])
        laps.append(
            {
                "lap": k + 1,
                "complete": True,
                "start_ns": s_ns,
                "end_ns": e_ns,
                "duration_s": (e_ns - s_ns) * 1e-9,
            }
        )
    return laps


def load_laps_json(path: Path) -> List[Dict]:
    data = json.loads(Path(path).read_text())
    return list(data["laps"])


def select_windows(
    *,
    mode: str,
    gt: Optional[pd.DataFrame],
    mcap_bounds: Tuple[int, int],
    laps: Optional[List[int]] = None,
    windows: Optional[List[Tuple[int, int]]] = None,
    laps_json: Optional[Path] = None,
) -> List[Dict]:
    """Resolve the list of run windows.

    Returns dicts: {name, start_ns, end_ns, lap(optional)}.
    """
    # explicit user windows take precedence
    if windows:
        return [
            {"name": f"win{idx}", "start_ns": int(s), "end_ns": int(e)}
            for idx, (s, e) in enumerate(windows)
        ]

    # external laps.json
    lap_rows: Optional[List[Dict]] = None
    if laps_json is not None:
        lap_rows = load_laps_json(Path(laps_json))
    elif mode == "eval" and gt is not None:
        lap_rows = detect_laps_from_gt(gt)

    if lap_rows:
        if laps:
            want = set(int(x) for x in laps)
            lap_rows = [L for L in lap_rows if int(L["lap"]) in want]
        return [
            {
                "name": f"lap{int(L['lap'])}",
                "lap": int(L["lap"]),
                "start_ns": int(L["start_ns"]),
                "end_ns": int(L["end_ns"]),
            }
            for L in lap_rows
        ]

    # fallback: whole mcap (predict default, or eval without detectable laps)
    lo, hi = mcap_bounds
    return [{"name": "full", "start_ns": int(lo), "end_ns": int(hi)}]
