"""Photometric prep for visual localization (map + query must match).

Pipeline: undistorted BGR -> gray0 (for highlight reject) + CLAHE on Lab L -> features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

PREPROCESS_VERSION = "clahe_highlight_v1"


@dataclass
class PhotometricConfig:
    enabled: bool = True
    clahe_clip: float = 2.5
    clahe_tile: int = 8
    highlight_mean_thr: float = 230.0
    highlight_sat_frac: float = 0.30
    bright_flat_mean: float = 180.0
    bright_flat_std: float = 12.0
    patch_radius: int = 7

    def to_meta(self) -> dict:
        d = asdict(self)
        d["version"] = PREPROCESS_VERSION
        return d


DEFAULT_PHOTOMETRIC = PhotometricConfig()


def clahe_bgr(bgr: np.ndarray, *, clip: float = 2.5, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def gray_from_bgr(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr.astype(np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _patch_stats(gray: np.ndarray, x: int, y: int, r: int) -> Tuple[float, float, float]:
    h, w = gray.shape[:2]
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(patch))
    std = float(np.std(patch))
    sat = float(np.mean(patch > 250))
    return mean, std, sat


def is_highlight_keypoint(
    gray0: np.ndarray,
    x: float,
    y: float,
    cfg: PhotometricConfig = DEFAULT_PHOTOMETRIC,
) -> bool:
    xi = int(round(x))
    yi = int(round(y))
    mean, std, sat = _patch_stats(gray0, xi, yi, cfg.patch_radius)
    if mean > cfg.highlight_mean_thr or sat > cfg.highlight_sat_frac:
        return True
    if mean > cfg.bright_flat_mean and std < cfg.bright_flat_std:
        return True
    return False


def prepare_feature_bgr(
    bgr_undist: np.ndarray,
    cfg: PhotometricConfig = DEFAULT_PHOTOMETRIC,
) -> Tuple[np.ndarray, np.ndarray]:
    gray0 = gray_from_bgr(bgr_undist)
    if cfg.enabled:
        bgr_feat = clahe_bgr(
            bgr_undist, clip=cfg.clahe_clip, tile=cfg.clahe_tile
        )
    else:
        bgr_feat = bgr_undist
    return bgr_feat, gray0


def filter_kpts_highlight(
    kpts: np.ndarray,
    desc: np.ndarray,
    scores: np.ndarray,
    gray0: np.ndarray,
    *,
    cfg: PhotometricConfig = DEFAULT_PHOTOMETRIC,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(kpts) == 0 or not cfg.enabled:
        return kpts, desc, scores
    xy = np.asarray(kpts, dtype=np.float32)
    keep = np.ones(len(xy), dtype=bool)
    for i, (x, y) in enumerate(xy):
        if is_highlight_keypoint(gray0, float(x), float(y), cfg):
            keep[i] = False
    D = desc if desc.ndim == 2 and desc.shape[0] == len(kpts) else np.asarray(desc).T
    return kpts[keep], D[keep], scores[keep]
