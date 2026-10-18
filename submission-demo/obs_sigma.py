"""Observation error std vs racing-line arc length s (from obs_error_sigma_s.py).

Default params: error-model/obs_error_sigma_vs_s.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_JSON = HERE / "error-model" / "obs_error_sigma_vs_s.json"

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
    """Observation error std [m] at racing-line s [m]."""
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
