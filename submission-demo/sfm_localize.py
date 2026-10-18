"""Online single-camera localization + query cache (from sfm_hfnet.py query half).

Loads a prebuilt sp_nv_nn map bundle, extracts query frames from the input
mcap for a given time window, runs features + PnP per frame, and writes
``errors.csv`` + ``query_meta.json``. Repeat runs with the same map, mcap,
window, stride and top_k reuse the cached ``errors.csv`` instead of recomputing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import sfm_core as sfm
import sfm_features as feat

QUERY_META_NAME = "query_meta.json"
LOC_TIMES_NAME = "localize_times_s.npy"
ERRORS_NAME = "errors.csv"

ERROR_COLS = [
    "stamp_ns",
    "ok",
    "gt_x",
    "gt_y",
    "gt_yaw",
    "pred_x",
    "pred_y",
    "pred_yaw",
    "err_xy",
    "err_yaw_deg",
    "n_pnp_corr",
    "n_pnp_inl",
]


def map_bundle_fingerprint(map_dir: Path) -> Dict:
    p = Path(map_dir) / sfm.MAP_BUNDLE_NAME
    if not p.is_file():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def mcap_fingerprint(mcap: Path) -> Dict:
    p = Path(mcap)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def loc_ready(loc_dir: Path) -> bool:
    d = Path(loc_dir)
    return (d / ERRORS_NAME).is_file() and (d / QUERY_META_NAME).is_file()


def query_cache_valid(
    loc_dir: Path,
    *,
    map_dir: Path,
    mcap: Path,
    window: Tuple[int, int],
    stride: int,
    top_k: int,
    backend: str,
) -> bool:
    """True iff cached query matches current map + mcap + window + params."""
    if not loc_ready(loc_dir):
        return False
    try:
        meta = json.loads((Path(loc_dir) / QUERY_META_NAME).read_text())
    except Exception:
        return False
    fp_map = map_bundle_fingerprint(map_dir)
    fp_mcap = mcap_fingerprint(mcap)
    if not fp_map.get("exists") or not fp_mcap.get("exists"):
        return False
    cm = meta.get("map_fingerprint") or {}
    cd = meta.get("mcap_fingerprint") or {}
    if cm.get("mtime_ns") != fp_map.get("mtime_ns") or cm.get("size") != fp_map.get("size"):
        return False
    if cd.get("mtime_ns") != fp_mcap.get("mtime_ns") or cd.get("size") != fp_mcap.get("size"):
        return False
    if tuple(meta.get("window", (None, None))) != tuple(int(x) for x in window):
        return False
    if int(meta.get("query_stride", -1)) != int(stride):
        return False
    if int(meta.get("top_k", -1)) != int(top_k):
        return False
    if str(meta.get("backend", "")) != str(backend):
        return False
    return True


def localize_camera(
    cam: str,
    *,
    mcap: Path,
    window: Tuple[int, int],
    map_dir: Path,
    out_loc_dir: Path,
    feats,
    calib: sfm.CamCalib,
    gt: Optional[pd.DataFrame],
    backend: str = "sp_nv_nn",
    query_stride: int = 1,
    top_k: int = 5,
    min_nn_dist: float = 0.5,
    override_query: bool = False,
    polygons_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Localize one camera over [window]; cache + reuse errors.csv.

    Returns an errors DataFrame with ERROR_COLS. GT columns are NaN when
    ``gt`` is None (predict mode).
    """
    import time

    cam = cam.strip().lower()
    out_loc_dir = Path(out_loc_dir)
    out_loc_dir.mkdir(parents=True, exist_ok=True)
    start_ns, end_ns = int(window[0]), int(window[1])

    if not override_query and query_cache_valid(
        out_loc_dir,
        map_dir=map_dir,
        mcap=mcap,
        window=(start_ns, end_ns),
        stride=query_stride,
        top_k=top_k,
        backend=backend,
    ):
        print(f"[{cam}] cache hit -> {out_loc_dir / ERRORS_NAME}")
        return pd.read_csv(out_loc_dir / ERRORS_NAME)

    map_frames, points = sfm.load_map_bundle(map_dir)
    print(f"[{cam}] map frames={len(map_frames)} points={len(points)}")

    frames = sfm.extract_query_frames(
        mcap,
        start_ns,
        end_ns,
        calib,
        gt,
        camera=cam,
        stride=query_stride,
    )
    print(f"[{cam}] query frames={len(frames)} window=[{start_ns},{end_ns}]")

    ego_mask = feat.load_ego_mask(
        cam, polygons_path if polygons_path is not None else feat.DEFAULT_POLYGONS
    )
    feat.run_joint_features(
        frames, feats, min_nn_dist=min_nn_dist, ego_mask=ego_mask, desc=f"feat-{cam}"
    )

    loc_times = []
    rows = []
    t0 = time.perf_counter()
    for q in frames:
        t_i = time.perf_counter()
        r = sfm.localize_query(
            q, map_frames, points, feats, calib.K, calib.T_base_cam, top_k=top_k
        )
        loc_times.append(time.perf_counter() - t_i)
        pred = r["pred"]
        rows.append(
            {
                "stamp_ns": int(q.stamp_ns),
                "ok": bool(r["ok"]),
                "gt_x": q.gt_xyzyaw[0],
                "gt_y": q.gt_xyzyaw[1],
                "gt_yaw": q.gt_xyzyaw[3],
                "pred_x": pred[0] if (r["ok"] and pred is not None) else float("nan"),
                "pred_y": pred[1] if (r["ok"] and pred is not None) else float("nan"),
                "pred_yaw": pred[3] if (r["ok"] and pred is not None) else float("nan"),
                "err_xy": r["err_xy"] if r["err_xy"] is not None else float("nan"),
                "err_yaw_deg": r["err_yaw_deg"]
                if r["err_yaw_deg"] is not None
                else float("nan"),
                "n_pnp_corr": int(r["n_pnp_corr"]),
                "n_pnp_inl": int(r["n_pnp_inl"]),
            }
        )
    loc_total = time.perf_counter() - t0
    loc_times_a = np.asarray(loc_times, dtype=np.float64)
    df = pd.DataFrame(rows, columns=ERROR_COLS)

    df.to_csv(out_loc_dir / ERRORS_NAME, index=False)
    df.loc[df.ok, ["stamp_ns", "pred_x", "pred_y", "pred_yaw"]].to_csv(
        out_loc_dir / "submission_xy.csv", index=False
    )
    np.save(out_loc_dir / LOC_TIMES_NAME, loc_times_a)

    meta = {
        "camera": cam,
        "backend": backend,
        "window": [start_ns, end_ns],
        "query_stride": int(query_stride),
        "top_k": int(top_k),
        "map_dir": str(Path(map_dir).resolve()),
        "map_fingerprint": map_bundle_fingerprint(map_dir),
        "mcap_fingerprint": mcap_fingerprint(mcap),
        "n_query": int(len(df)),
        "n_success": int(df.ok.astype(bool).sum()),
        "localize_total_s": float(loc_total),
        "runtime_ms_mean": float(np.mean(loc_times_a) * 1000) if len(loc_times_a) else None,
    }
    (out_loc_dir / QUERY_META_NAME).write_text(json.dumps(meta, indent=2))
    print(
        f"[{cam}] localized {len(df)} ok={meta['n_success']} "
        f"in {loc_total:.1f}s -> {out_loc_dir / ERRORS_NAME}"
    )
    return df
