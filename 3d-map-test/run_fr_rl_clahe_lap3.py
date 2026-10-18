#!/usr/bin/env python3
"""Build fr/rl Lap2+4 maps (CLAHE+highlight) + Lap3 query + se2 fuse; copy maps to submission-demo."""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
SFM = ROOT / "3d-map-test"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SFM))

import sfm_experiment as sfm
import sfm_hfnet as hf

sfm = importlib.reload(sfm)
hf = importlib.reload(hf)

OUT = SFM / "outputs_hfnet_clahe_fr_rl"
POLYGONS = ROOT / "fr-auto-mask" / "polygons.json"
MAP_MCAP = (
    ROOT.parent
    / "c0_2026-10-13_22-53-42_camloc"
    / "23-00-38_cut_120m_update"
    / "map.mcap"
)
SUBMIT_MAP = ROOT / "submission-demo" / "sfm-model"

CAMERAS = ("fr", "rl")
MAP_LAPS = (2, 4)
QUERY_LAP = 3
REF_CAMERA = "fr"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = hf.require_mps()
    backend, is_proxy = hf.resolve_hfnet_backend(None)
    print("device", device, "backend", backend, "proxy", is_proxy)
    print("preprocess", hf.ipre.PREPROCESS_VERSION)

    gt = sfm.load_gt()
    laps_map = {L: sfm.load_lap(L) for L in MAP_LAPS}
    lap_q = sfm.load_lap(QUERY_LAP)
    calibs = {c: sfm.load_camera_calib(c) for c in CAMERAS}
    track = hf.load_racing_line(MAP_MCAP)["track"]
    feats = hf.make_features("sp_nv_nn", max_keypoints=1024, resize=512, ratio=0.8)

    runs = {}
    for cam in CAMERAS:
        print("\n" + "=" * 60, f"CAMERA {cam}", "=" * 60)
        runs[cam] = hf.run_camera_multi_lap_s_pipeline(
            cam,
            feats=feats,
            calib=calibs[cam],
            gt=gt,
            laps_map=laps_map,
            lap_q=lap_q,
            track_line=track,
            out_root=OUT,
            map_laps=MAP_LAPS,
            query_lap=QUERY_LAP,
            map_max_frames=None,
            query_max_frames=None,
            map_stride=1,
            query_stride=1,
            min_nn_dist=0.50,
            neighbor_mode="arc",
            neighbor_k=6,
            neighbor_offset=3,
            n_max=None,
            min_baseline_m=0.5,
            epi_thresh_px=6.0,
            top_k=5,
            override_map=True,
            override_query=True,
            polygons_path=POLYGONS,
            map_mcap=MAP_MCAP,
        )
        m = runs[cam]["metrics"]
        print(
            f"[{cam}] success={m['n_success']}/{m['n_query']}  "
            f"A-RMSE={m['pos_rmse_xy_m']:.3f}  points={runs[cam]['n_points']}"
        )

    tables = {c: runs[c]["errors_df"] for c in CAMERAS}
    fuse_dir = OUT / f"loc_lap{QUERY_LAP}_fused"
    fuse_dir.mkdir(parents=True, exist_ok=True)
    df_se2 = hf.fuse_multi_camera_tables(
        tables, ref_camera=REF_CAMERA, mode="se2"
    )
    df_se2.to_csv(fuse_dir / "errors_se2.csv", index=False)
    print(
        f"fuse[se2] success {df_se2.ok.sum()}/{len(df_se2)}  "
        f"modes={df_se2.fuse_mode.value_counts().to_dict()}"
    )

    run_meta = {
        "backend": backend,
        "proxy": is_proxy,
        "preprocess": hf.ipre.DEFAULT_PHOTOMETRIC.to_meta(),
        "cameras": list(CAMERAS),
        "map_laps": list(MAP_LAPS),
        "query_lap": QUERY_LAP,
        "ref_camera": REF_CAMERA,
        "per_camera": {
            c: {
                "n_points": runs[c]["n_points"],
                "n_pairs": runs[c]["n_pairs"],
                "metrics": runs[c]["metrics"],
            }
            for c in CAMERAS
        },
    }
    (OUT / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    SUBMIT_MAP.mkdir(parents=True, exist_ok=True)
    for cam in CAMERAS:
        src = OUT / f"map_{cam}_lap2+4"
        dst = SUBMIT_MAP / f"map_{cam}_lap2+4"
        if dst.is_dir():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print("copied map ->", dst)
    print("done OUT", OUT)


if __name__ == "__main__":
    main()
