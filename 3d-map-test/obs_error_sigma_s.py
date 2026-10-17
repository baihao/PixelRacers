"""Observation error std vs racing-line arc length s.

Default parameters loaded from:
  outputs_hfnet_multi_lap_s/loc_lap3_fused/obs_error_sigma_vs_s.json

Usage:
  from obs_error_sigma_s import sigma_of_s, load_params
  sigma_of_s(1200.0)          # meters
  sigma_of_s([0, 100, 500])   # ndarray
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

DEFAULT_JSON = (
    Path(__file__).resolve().parent
    / "outputs_hfnet_multi_lap_s"
    / "loc_lap3_fused"
    / "obs_error_sigma_vs_s.json"
)

_PARAMS = None


def load_params(path: Optional[Path] = None) -> dict:
    global _PARAMS
    path = Path(path) if path is not None else DEFAULT_JSON
    _PARAMS = json.loads(Path(path).read_text())
    return _PARAMS


def _circ_dist(a, b, lap: float):
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return np.minimum(d, lap - d)


def sigma_of_s(
    s: Union[float, np.ndarray],
    params: Optional[dict] = None,
) -> Union[float, np.ndarray]:
    """Evaluate observation error standard deviation [m] at racing-line s [m]."""
    if params is None:
        params = _PARAMS if _PARAMS is not None else load_params()
    L = float(params["L_m"])
    sigma0 = float(params["sigma0_m"])
    peaks = params["peaks"]
    scalar = np.ndim(s) == 0
    sq = np.mod(np.atleast_1d(np.asarray(s, dtype=float)), L)
    out = np.full(sq.shape, sigma0, dtype=float)
    for p in peaks:
        out = out + float(p["a_m"]) * np.exp(
            -0.5 * (_circ_dist(sq, float(p["mu_m"]), L) / float(p["w_m"])) ** 2
        )
    return float(out[0]) if scalar else out


if __name__ == "__main__":
    load_params()
    print("L", _PARAMS["L_m"], "K", _PARAMS["K"], "sigma0", _PARAMS["sigma0_m"])
    for s in (0.0, 500.0, 1500.0, 2500.0):
        print(f"  sigma({s:.0f}) = {sigma_of_s(s):.4f} m")
