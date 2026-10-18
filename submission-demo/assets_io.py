"""Extract + load mcap assets (calib / tf / track_bounds / state_estimation / gt).

Resources are pulled straight out of the input .mcap using the scripts in
``tools/`` (``extract_assets.py``, ``extract_gt.py``) and cached under an
``assets/`` directory so repeat runs skip re-extraction.

- ``extract_assets`` -> camera_info.json, tf_static.json, track_bounds.json,
  state_estimation.csv   (available for BOTH training and test mcaps)
- ``extract_gt``     -> gt.csv (from /tf; training only — test has no GT)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

HERE = Path(__file__).resolve().parent
TOOLS = HERE / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def _read_csv_ns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"timestamp_ns": str})
    df["timestamp_ns"] = df["timestamp_ns"].astype("int64")
    return df


def extract_assets(mcap: Path | str, out_dir: Path | str, *, force: bool = False) -> Path:
    """Run tools/extract_assets.py -> camera_info/tf_static/track_bounds/state_estimation."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    needed = [
        out / "camera_info.json",
        out / "tf_static.json",
        out / "track_bounds.json",
        out / "state_estimation.csv",
    ]
    if force or not all(p.is_file() for p in needed):
        import extract_assets as _ea  # tools/extract_assets.py

        _ea.main(str(mcap), str(out))
    return out


def extract_gt(mcap: Path | str, out_dir: Path | str, *, force: bool = False) -> Path:
    """Run tools/extract_gt.py -> gt.csv (training only)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gt_path = out / "gt.csv"
    if force or not gt_path.is_file():
        import extract_gt as _eg  # tools/extract_gt.py

        _eg.main(str(mcap), str(gt_path))
    return gt_path


@dataclass
class Assets:
    dir: Path
    camera_info: dict
    tf_static: dict
    track_bounds: dict
    se: pd.DataFrame
    gt: Optional[pd.DataFrame] = None

    @property
    def has_gt(self) -> bool:
        return self.gt is not None and len(self.gt) > 0


def ensure_assets(
    mcap: Path | str,
    out_dir: Path | str,
    *,
    want_gt: bool,
    force: bool = False,
) -> Assets:
    """Extract (if missing) and load all mcap assets into memory."""
    out = Path(out_dir)
    extract_assets(mcap, out, force=force)
    gt_df: Optional[pd.DataFrame] = None
    if want_gt:
        extract_gt(mcap, out, force=force)
        gt_df = _read_csv_ns(out / "gt.csv")

    camera_info = json.loads((out / "camera_info.json").read_text())
    tf_static = json.loads((out / "tf_static.json").read_text())
    track_bounds = json.loads((out / "track_bounds.json").read_text())
    se = _read_csv_ns(out / "state_estimation.csv")
    return Assets(
        dir=out,
        camera_info=camera_info,
        tf_static=tf_static,
        track_bounds=track_bounds,
        se=se,
        gt=gt_df,
    )
