"""Ensure SfM loc_lap{N}_{cam} exists for query-demo (reuse map_*_lap2+4)."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SFM = ROOT / "3d-map-test"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from query_demo_ekf import DEFAULT_LOC_CAMERAS, SFM_OUT

OUT = SFM_OUT
POLYGONS = ROOT / "fr-auto-mask" / "polygons.json"
MAP_MCAP = (
    ROOT.parent
    / "c0_2026-10-13_22-53-42_camloc"
    / "23-00-38_cut_120m_update"
    / "map.mcap"
)


def _setup_path() -> None:
    for p in (ROOT, SFM):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def loc_ready(
    query_lap: int, cameras: Sequence[str] = DEFAULT_LOC_CAMERAS
) -> bool:
    return all(
        (OUT / f"loc_lap{int(query_lap)}_{c}" / "errors.csv").is_file() for c in cameras
    )


def ensure_lap_loc(
    query_lap: int = 5,
    *,
    cameras: Sequence[str] = DEFAULT_LOC_CAMERAS,
    map_laps: Tuple[int, ...] = (2, 4),
    override_query: bool = False,
    override_map: bool = False,
    query_stride: int = 1,
    query_max_frames: int | None = None,
) -> Dict:
    """Run multi-lap s-order query for ``query_lap`` if cache missing."""
    _setup_path()
    import sfm_experiment as sfm
    import sfm_hfnet as hf

    sfm = importlib.reload(sfm)
    hf = importlib.reload(hf)

    OUT.mkdir(parents=True, exist_ok=True)
    if loc_ready(query_lap, cameras) and not override_query:
        return {
            "skipped": True,
            "query_lap": query_lap,
            "paths": {
                c: str(OUT / f"loc_lap{query_lap}_{c}" / "errors.csv") for c in cameras
            },
        }

    if query_lap in map_laps:
        raise ValueError(f"query_lap={query_lap} must not be in map_laps={map_laps}")

    device = hf.require_mps()
    backend, is_proxy = hf.resolve_hfnet_backend(None)
    print("device", device, "backend", backend, "proxy", is_proxy)

    gt = sfm.load_gt()
    laps_map = {L: sfm.load_lap(L) for L in map_laps}
    lap_q = sfm.load_lap(query_lap)
    calibs = {c: sfm.load_camera_calib(c) for c in cameras}
    track = hf.load_racing_line(MAP_MCAP)["track"]

    if backend == "hfnet_onnx":
        feats = hf.make_features(
            "hfnet_onnx",
            onnx_path=hf.DEFAULT_ONNX,
            max_keypoints=1024,
            resize=512,
            ratio=0.8,
        )
    else:
        feats = hf.make_features(
            "sp_nv_nn", max_keypoints=1024, resize=512, ratio=0.8
        )

    summaries = {}
    for cam in cameras:
        print("\n" + "=" * 60)
        print("CAMERA", cam, "query_lap", query_lap)
        print("=" * 60)
        summaries[cam] = hf.run_camera_multi_lap_s_pipeline(
            cam,
            feats=feats,
            calib=calibs[cam],
            gt=gt,
            laps_map=laps_map,
            lap_q=lap_q,
            track_line=track,
            out_root=OUT,
            map_laps=map_laps,
            query_lap=query_lap,
            map_max_frames=None,
            query_max_frames=query_max_frames,
            map_stride=1,
            query_stride=query_stride,
            min_nn_dist=0.50,
            neighbor_mode="arc",
            neighbor_k=6,
            neighbor_offset=3,
            n_max=None,
            min_baseline_m=0.5,
            epi_thresh_px=6.0,
            top_k=5,
            override_map=override_map,
            override_query=override_query,
            polygons_path=POLYGONS,
            map_mcap=MAP_MCAP,
        )
        m = summaries[cam]["metrics"]
        print(
            f"[{cam}] success={m['n_success']}/{m['n_query']}  "
            f"A-RMSE={m['pos_rmse_xy_m']:.3f}"
        )

    return {"skipped": False, "query_lap": query_lap, "summaries": summaries}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lap", type=int, default=5)
    ap.add_argument("--override-query", action="store_true")
    ap.add_argument("--override-map", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    out = ensure_lap_loc(
        args.lap,
        override_query=args.override_query,
        override_map=args.override_map,
        query_stride=args.stride,
        query_max_frames=args.max_frames,
    )
    print(out if out.get("skipped") else f"done query lap {args.lap}")


if __name__ == "__main__":
    main()
