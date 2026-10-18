#!/usr/bin/env python3
"""CLI wrapper around pipeline.run_all.

Examples:
  # eval on training mcap, all laps, all three options
  python run_submission.py --mode eval --train-mcap /path/c0_..._0.mcap

  # eval only lap 3, EKF option
  python run_submission.py --mode eval --train-mcap ... --laps 3 --options ekf

  # predict on test mcap (whole mcap, submission only)
  python run_submission.py --mode predict --test-mcap /path/test.mcap
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")  # headless: save figures, never block

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pipeline import SubmissionConfig, run_all


def _parse_windows(vals):
    if not vals:
        return None
    out = []
    for v in vals:
        s, e = v.split(":")
        out.append((int(s), int(e)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Submission pipeline (sp_nv_nn backend)")
    ap.add_argument("--mode", choices=["eval", "predict"], default="eval")
    ap.add_argument("--train-mcap", default=None)
    ap.add_argument("--test-mcap", default=None)
    ap.add_argument("--laps", type=int, nargs="*", default=None, help="lap numbers; default all")
    ap.add_argument("--windows", nargs="*", default=None,
                    help="manual start:end ns windows (predict)")
    ap.add_argument("--laps-json", default=None, help="external laps.json for windows")
    ap.add_argument("--cams", nargs="*", default=["fr", "rl"])
    ap.add_argument("--ref-camera", default="fr")
    ap.add_argument("--backend", default="sp_nv_nn")
    ap.add_argument("--query-stride", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--override-query", action="store_true")
    ap.add_argument("--options", nargs="*", default=["motion_only", "two_cam", "ekf"],
                    choices=["motion_only", "two_cam", "ekf"])
    args = ap.parse_args(argv)

    cfg = SubmissionConfig(
        train_mcap=args.train_mcap,
        test_mcap=args.test_mcap,
        mode=args.mode,
        laps=args.laps if args.laps else None,
        windows=_parse_windows(args.windows),
        laps_json=args.laps_json,
        cams=tuple(args.cams),
        ref_camera=args.ref_camera,
        backend=args.backend,
        query_stride=args.query_stride,
        top_k=args.top_k,
        override_query=args.override_query,
    )
    run_all(cfg, options=tuple(args.options))


if __name__ == "__main__":
    main()
