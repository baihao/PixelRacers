"""Feature backends + ego-mask helpers (from sfm_hfnet.py).

Main backend: ``sp_nv_nn`` (SuperPoint local + NetVLAD global, mutual-NN match)
— matches the prebuilt ``map_*_lap2+4`` bundles. ``HFNetONNX`` is kept as a
future-upgrade stub (needs ``onnxruntime`` + an ``hfnet.onnx`` weight file and
a map rebuilt with HF-Net descriptors).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

import sfm_core as sfm

HERE = Path(__file__).resolve().parent
DEFAULT_POLYGONS = HERE / "sfm-model" / "polygons.json"
DEFAULT_ONNX = HERE / "sfm-model" / "models" / "hfnet.onnx"


def mutual_nn_match(
    k0: np.ndarray,
    d0: np.ndarray,
    k1: np.ndarray,
    d1: np.ndarray,
    *,
    ratio: float = 0.8,
    cross_check: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mutual nearest-neighbor + Lowe ratio on L2-normalized descriptors."""
    if len(k0) == 0 or len(k1) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    D0 = d0 if d0.ndim == 2 and d0.shape[0] == len(k0) else d0.T
    D1 = d1 if d1.ndim == 2 and d1.shape[0] == len(k1) else d1.T
    D0 = D0.astype(np.float32)
    D1 = D1.astype(np.float32)
    sim = D0 @ D1.T
    nn1 = np.argmax(sim, axis=1)
    s1 = sim[np.arange(len(D0)), nn1]
    sim_masked = sim.copy()
    sim_masked[np.arange(len(D0)), nn1] = -1.0
    s2 = sim_masked.max(axis=1)
    dist1 = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * s1))
    dist2 = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * s2))
    ratio_ok = dist1 <= (ratio * dist2 + 1e-6)
    idx0 = np.where(ratio_ok)[0]
    idx1 = nn1[ratio_ok]
    if cross_check and len(idx0):
        nn0_from1 = np.argmax(sim, axis=0)
        keep = nn0_from1[idx1] == idx0
        idx0, idx1 = idx0[keep], idx1[keep]
    return k0[idx0].astype(np.float32), k1[idx1].astype(np.float32)


class NNMatcher:
    """mutual-NN matcher. Exposes match() consumed by sfm_core.localize_query."""

    def __init__(self, ratio: float = 0.8):
        self.ratio = ratio

    def match(self, k0, d0, k1, d1, size0, size1):
        return mutual_nn_match(k0, d0, k1, d1, ratio=self.ratio)


class SPNVJointFeatures(NNMatcher):
    """sp_nv_nn backend: SuperPoint local + NetVLAD global, joint extract API."""

    def __init__(
        self,
        device: Optional[str] = None,
        max_keypoints: int = 1024,
        resize: int = 512,
        ratio: float = 0.8,
    ):
        super().__init__(ratio=ratio)
        self.device = sfm.pick_device(device)
        self.resize = resize
        self.sp = sfm.LocalFeatures(
            device=self.device, max_keypoints=max_keypoints, resize=resize
        )
        self.netvlad = sfm.NetVLADGlobal(device=self.device)

    def extract_joint(
        self, bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kpts, desc, scores = self.sp.extract(bgr)
        g = self.netvlad.extract(bgr)
        return kpts, desc, scores, g


class HFNetONNX(NNMatcher):
    """Future upgrade: real HF-Net (joint heads) via ONNX Runtime.

    Not used in this version. Requires sfm-model/models/hfnet.onnx and a map
    rebuilt with HF-Net descriptors. Kept so make_features("hfnet_onnx") works
    once those resources exist.
    """

    def __init__(
        self,
        onnx_path: Path = DEFAULT_ONNX,
        device: Optional[str] = None,
        max_keypoints: int = 1024,
        resize: int = 640,
        ratio: float = 0.8,
        keypoint_threshold: float = 0.005,
    ):
        super().__init__(ratio=ratio)
        import onnxruntime as ort  # noqa: F401

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"HF-Net ONNX not found: {self.onnx_path}\n"
                f"See {DEFAULT_ONNX.parent / 'README_HFNET.md'}"
            )
        self.device = sfm.pick_device(device)
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "mps":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.resize = resize
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self._input_name = self.session.get_inputs()[0].name
        self._outputs = [o.name for o in self.session.get_outputs()]

    def extract_joint(self, bgr: np.ndarray):  # pragma: no cover - future use
        raise NotImplementedError(
            "HFNetONNX.extract_joint is a stub for a future HF-Net upgrade. "
            "This version uses the sp_nv_nn backend."
        )


def make_features(
    backend: str = "sp_nv_nn",
    **kwargs,
) -> Union[SPNVJointFeatures, HFNetONNX]:
    backend = backend.lower().strip()
    if backend in ("hfnet", "hfnet_onnx", "onnx"):
        return HFNetONNX(**kwargs)
    if backend in ("sp_nv_nn", "proxy", "sp+nv"):
        return SPNVJointFeatures(**kwargs)
    raise ValueError(f"unknown backend {backend!r}; use sp_nv_nn or hfnet_onnx")


def resolve_backend(preferred: Optional[str] = None) -> str:
    """Resolve feature backend. Defaults to sp_nv_nn (this version's main path)."""
    if preferred:
        b = preferred.lower().strip()
        if b in ("hfnet", "hfnet_onnx", "onnx"):
            if not DEFAULT_ONNX.is_file():
                raise FileNotFoundError(
                    f"hfnet_onnx requested but weight missing: {DEFAULT_ONNX}"
                )
            return "hfnet_onnx"
        if b in ("sp_nv_nn", "proxy", "sp+nv"):
            return "sp_nv_nn"
        raise ValueError(f"unknown backend {preferred!r}")
    return "sp_nv_nn"


# ---------------------------------------------------------------------------
# ego mask
# ---------------------------------------------------------------------------
def load_ego_mask(
    camera: str,
    polygons_path: Path = DEFAULT_POLYGONS,
    *,
    out_hw: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """uint8 mask: 255=allowed, 0=ego/ignored (from polygons.json)."""
    cam = camera.strip().lower()
    polygons_path = Path(polygons_path)
    if not polygons_path.is_file():
        return None
    data = json.loads(polygons_path.read_text())
    polys = data.get("masked_polygons", {}).get(cam, [])
    if not polys:
        return None
    w0 = int(data["image_size"]["width"])
    h0 = int(data["image_size"]["height"])
    mask = np.full((h0, w0), 255, dtype=np.uint8)
    for poly in polys:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 0)
    if out_hw is not None:
        h, w = out_hw
        if (h, w) != (h0, w0):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def apply_ego_mask_bgr(bgr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return bgr
    if mask.shape[:2] != bgr.shape[:2]:
        mask = cv2.resize(
            mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    out = bgr.copy()
    out[mask == 0] = 0
    return out


def filter_kpts_by_mask(
    kpts: np.ndarray,
    desc: np.ndarray,
    scores: np.ndarray,
    mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mask is None or len(kpts) == 0:
        return kpts, desc, scores
    xy = np.asarray(kpts, dtype=np.float32)
    xs = np.clip(np.round(xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(np.round(xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
    keep = mask[ys, xs] > 0
    D = desc if desc.ndim == 2 and desc.shape[0] == len(kpts) else np.asarray(desc).T
    return kpts[keep], D[keep], scores[keep]


def run_joint_features(
    frames,
    feats,
    *,
    min_nn_dist: float = 0.5,
    ego_mask: Optional[np.ndarray] = None,
    desc: str = "features",
) -> None:
    """Extract local+global features for each frame (in place)."""
    try:
        from tqdm.auto import tqdm

        it = tqdm(frames, desc=desc, unit="frm")
    except Exception:
        it = frames
    for fr in it:
        img = apply_ego_mask_bgr(fr.bgr_undist, ego_mask)
        kpts, d, scores, g = feats.extract_joint(img)
        kpts, d, scores = filter_kpts_by_mask(kpts, d, scores, ego_mask)
        kpts, d, scores = sfm.filter_ambiguous_descriptors(
            kpts, d, scores, min_nn_dist=min_nn_dist
        )
        fr.kpts, fr.desc, fr.scores = kpts, d, scores
        fr.global_desc = g
