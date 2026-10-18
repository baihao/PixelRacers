"""Self-contained SfM localize core (subset of sfm_experiment.py).

Provides: camera calibration (from extracted assets dir), frame extraction +
undistort, SuperPoint local features, NetVLAD global descriptor, 3D map-point
projection, NetVLAD top-K retrieval, PnP localization, and map-bundle load.

Backend note: this version uses the ``sp_nv_nn`` feature stack (SuperPoint +
NetVLAD + mutual-NN), matching the prebuilt ``map_*_lap2+4`` bundles.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
THIRD_PARTY = HERE.parent / "third_party" / "Hierarchical-Localization"

MAP_BUNDLE_NAME = "map_bundle.npz"


# ---------------------------------------------------------------------------
# device + geometry helpers
# ---------------------------------------------------------------------------
def pick_device(device: Optional[str] = None) -> str:
    import torch

    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def T_from_tf(tf: dict, key: str) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot

    d = tf[key]
    t = np.array([d["translation"][k] for k in "xyz"], dtype=np.float64)
    q = d["rotation"]
    Rm = Rot.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return T


def T_map_base(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [x, y, z]
    return T


def pose_xyzyaw(T: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(T[0, 3]),
        float(T[1, 3]),
        float(T[2, 3]),
        float(np.arctan2(T[1, 0], T[0, 0])),
    )


def T_map_cam(gt_pose, T_base_cam: np.ndarray) -> np.ndarray:
    return T_map_base(*gt_pose) @ T_base_cam


def relative_T(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T_a) @ T_b


def undistort(bgr: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    return cv2.undistort(bgr, K, D, None, K)


# ---------------------------------------------------------------------------
# calibration from extracted assets dir
# ---------------------------------------------------------------------------
@dataclass
class CamCalib:
    K: np.ndarray
    D: np.ndarray
    T_base_cam: np.ndarray


def load_camera_calib(camera: str, assets_dir: Path) -> CamCalib:
    """Load K, D, T_base_cam from assets_dir/{camera_info.json, tf_static.json}."""
    cam = camera.strip().lower()
    assets_dir = Path(assets_dir)
    ci = json.loads((assets_dir / "camera_info.json").read_text())
    tf = json.loads((assets_dir / "tf_static.json").read_text())
    if cam not in ci:
        raise KeyError(f"camera_info.json missing key {cam!r}; have {list(ci)}")
    tf_key = f"constructor0/base_link->constructor0/camera_{cam}"
    if tf_key not in tf:
        raise KeyError(f"tf_static.json missing {tf_key!r}")
    return CamCalib(
        K=np.array(ci[cam]["K"], dtype=np.float64).reshape(3, 3),
        D=np.array(ci[cam]["D"], dtype=np.float64),
        T_base_cam=T_from_tf(tf, tf_key),
    )


def gt_at(gt: pd.DataFrame, ts: int) -> Tuple[float, float, float, float]:
    i = int(np.argmin(np.abs(gt["timestamp_ns"].to_numpy(np.int64) - np.int64(ts))))
    r = gt.iloc[i]
    return float(r.x), float(r.y), float(r.z), float(r.yaw)


# ---------------------------------------------------------------------------
# frame container
# ---------------------------------------------------------------------------
@dataclass
class FrameData:
    stamp_ns: int
    bgr_undist: np.ndarray
    T_map_cam: np.ndarray
    gt_xyzyaw: Tuple[float, float, float, float]
    kpts: Optional[np.ndarray] = None
    desc: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    global_desc: Optional[np.ndarray] = None


_NAN_POSE = (float("nan"), float("nan"), float("nan"), float("nan"))


def extract_query_frames(
    mcap: Path,
    start_ns: int,
    end_ns: int,
    calib: CamCalib,
    gt: Optional[pd.DataFrame],
    *,
    camera: str,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> List[FrameData]:
    """Decode + undistort camera frames in [start,end]; attach GT pose if available.

    When ``gt`` is None (predict / test), gt_xyzyaw is NaN and T_map_cam is
    identity — neither is used by localize_query's PnP output (only err_xy,
    which becomes NaN).
    """
    from mcap_io import iter_camera_frames

    cam = camera.strip().lower()
    frames: List[FrameData] = []
    it = iter_camera_frames(
        mcap, cameras=(cam,), start_ns=start_ns, end_ns=end_ns, max_frames=None
    )
    for i, fr in enumerate(it):
        if stride > 1 and i % stride != 0:
            continue
        und = undistort(fr.bgr, calib.K, calib.D)
        if gt is not None:
            gpose = gt_at(gt, fr.stamp_ns)
            Tmc = T_map_cam(gpose, calib.T_base_cam)
        else:
            gpose = _NAN_POSE
            Tmc = np.eye(4)
        frames.append(
            FrameData(
                stamp_ns=fr.stamp_ns,
                bgr_undist=und,
                T_map_cam=Tmc,
                gt_xyzyaw=gpose,
            )
        )
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


# ---------------------------------------------------------------------------
# SuperPoint local features + NetVLAD global descriptor
# ---------------------------------------------------------------------------
class LocalFeatures:
    """SuperPoint extractor (LightGlue package). match() unused in sp_nv_nn."""

    def __init__(
        self,
        device: Optional[str] = None,
        max_keypoints: int = 1024,
        resize: int = 512,
    ):
        import torch
        from lightglue import SuperPoint

        self.device = pick_device(device)
        self.resize = resize
        self._torch = torch
        self.extractor = (
            SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        )

    def extract(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from lightglue.utils import numpy_image_to_torch, rbd

        with self._torch.no_grad():
            feats = self.extractor.extract(
                numpy_image_to_torch(bgr).to(self.device), resize=self.resize
            )
        feats = rbd(feats)
        kpts = feats["keypoints"].detach().cpu().numpy()
        desc = feats["descriptors"].detach().cpu().numpy()
        scores = feats["keypoint_scores"].detach().cpu().numpy()
        return kpts, desc, scores


class NetVLADGlobal:
    """hloc NetVLAD global descriptor (downloads Pitts30K weights on first use)."""

    def __init__(self, device: Optional[str] = None):
        import sys
        import torch

        if str(THIRD_PARTY) not in sys.path:
            sys.path.insert(0, str(THIRD_PARTY))
        from hloc.extractors.netvlad import NetVLAD

        self._torch = torch
        self.device = pick_device(device)
        self.model = (
            NetVLAD({"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}).eval()
        )
        self.model = self.model.to(self.device)

    def extract(self, bgr: np.ndarray) -> np.ndarray:
        torch = self._torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        h, w = img.shape[-2:]
        scale = 1024 / max(h, w)
        if scale < 1:
            img = torch.nn.functional.interpolate(
                img[None], scale_factor=scale, mode="bilinear", align_corners=False
            )[0]
        with torch.no_grad():
            out = self.model({"image": img[None].to(self.device)})
        g = out["global_descriptor"][0].detach().cpu().numpy().astype(np.float32)
        g = g / (np.linalg.norm(g) + 1e-12)
        return g


def filter_ambiguous_descriptors(
    kpts: np.ndarray,
    desc: np.ndarray,
    scores: np.ndarray,
    *,
    min_nn_dist: float = 0.5,
    max_keep: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop keypoints whose descriptor is too similar to another in the image."""
    n = len(kpts)
    if n < 2:
        return kpts, desc, scores
    D = desc if desc.ndim == 2 and desc.shape[0] == n else desc.T
    sim = D @ D.T
    np.fill_diagonal(sim, -1.0)
    nn_sim = sim.max(axis=1)
    nn_dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * nn_sim))
    keep = nn_dist >= min_nn_dist
    kpts, D, scores = kpts[keep], D[keep], scores[keep]
    if max_keep is not None and len(kpts) > max_keep:
        order = np.argsort(-scores)[:max_keep]
        kpts, D, scores = kpts[order], D[order], scores[order]
    return kpts, D, scores


# ---------------------------------------------------------------------------
# localization
# ---------------------------------------------------------------------------
def project_points(
    K: np.ndarray, T_map_cam: np.ndarray, X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    Xc = (np.linalg.inv(T_map_cam) @ np.c_[X, np.ones(len(X))].T).T
    z = Xc[:, 2]
    uv = (K @ Xc[:, :3].T).T
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-9)
    ok = z > 0.5
    return uv, ok


def retrieve_topk(
    query_g: np.ndarray, db_g: np.ndarray, k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    sims = db_g @ query_g
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


def localize_query(
    query: FrameData,
    map_frames: List[FrameData],
    points: np.ndarray,
    matcher,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    *,
    top_k: int = 5,
    pnp_reproj: float = 4.0,
    min_inl: int = 12,
) -> Dict[str, Any]:
    """NetVLAD top-K retrieve -> mutual-NN match -> 2D-3D -> PnP RANSAC.

    ``matcher`` must expose ``match(k0, d0, k1, d1, size0, size1) -> (p0, p1)``.
    GT (query.gt_xyzyaw) may be NaN (predict); only err_* become NaN then.
    """
    from scipy.spatial import cKDTree

    db_g = np.stack([f.global_desc for f in map_frames], 0)
    idx, sims = retrieve_topk(query.global_desc, db_g, k=top_k)
    h, w = query.bgr_undist.shape[:2]
    size = (w, h)

    obj_list, img_list = [], []
    cand_stats = []
    for ci in idx:
        c = map_frames[int(ci)]
        p_q, p_c = matcher.match(query.kpts, query.desc, c.kpts, c.desc, size, size)
        if len(p_q) < 8:
            cand_stats.append({"frame": int(ci), "n_match": len(p_q), "n_3d": 0})
            continue
        uv_proj, okp = project_points(K, c.T_map_cam, points)
        if okp.sum() < 5:
            cand_stats.append({"frame": int(ci), "n_match": len(p_q), "n_3d": 0})
            continue
        tree = cKDTree(uv_proj[okp])
        dist, nn = tree.query(p_c, distance_upper_bound=3.0)
        valid = np.isfinite(dist)
        X = points[okp][nn[valid]]
        uq = p_q[valid]
        obj_list.append(X)
        img_list.append(uq)
        cand_stats.append(
            {
                "frame": int(ci),
                "n_match": int(len(p_q)),
                "n_3d": int(valid.sum()),
                "sim": float(sims[np.where(idx == ci)[0][0]]),
            }
        )

    out: Dict[str, Any] = {
        "topk": idx.tolist(),
        "sims": sims.tolist(),
        "cand_stats": cand_stats,
        "ok": False,
        "pred": None,
        "err_xy": None,
        "err_yaw_deg": None,
        "n_pnp_corr": 0,
        "n_pnp_inl": 0,
    }
    if not obj_list:
        return out
    obj = np.concatenate(obj_list, 0)
    img = np.concatenate(img_list, 0)
    out["n_pnp_corr"] = int(len(obj))
    if len(obj) < min_inl:
        return out
    okp, rvec, tvec, inl = cv2.solvePnPRansac(
        obj.astype(np.float64),
        img.astype(np.float64),
        K,
        None,
        reprojectionError=pnp_reproj,
        iterationsCount=500,
        confidence=0.999,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not okp or inl is None or len(inl) < min_inl:
        return out
    Rm, _ = cv2.Rodrigues(rvec)
    T_cam_map = np.eye(4)
    T_cam_map[:3, :3] = Rm
    T_cam_map[:3, 3] = tvec.ravel()
    T_map_cam_est = np.linalg.inv(T_cam_map)
    T_map_base = T_map_cam_est @ np.linalg.inv(T_base_cam)
    pred = pose_xyzyaw(T_map_base)
    g = query.gt_xyzyaw
    out["ok"] = True
    out["pred"] = pred
    out["n_pnp_inl"] = int(len(inl))
    out["err_xy"] = float(np.hypot(pred[0] - g[0], pred[1] - g[1]))
    out["err_yaw_deg"] = float(
        ((pred[3] - g[3] + np.pi) % (2 * np.pi) - np.pi) * 180 / np.pi
    )
    return out


# ---------------------------------------------------------------------------
# map bundle load
# ---------------------------------------------------------------------------
def map_ready(map_dir: Path) -> bool:
    return (Path(map_dir) / MAP_BUNDLE_NAME).is_file()


def load_map_bundle(map_dir: Path) -> Tuple[List[FrameData], np.ndarray]:
    """Load map frames (features/poses, placeholder images) + 3D points."""
    path = Path(map_dir) / MAP_BUNDLE_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    points = np.asarray(z["points"], dtype=np.float64)
    stamps = z["stamps"]
    xyzyaw = z["xyzyaw"]
    Tmc = z["T_map_cam"]
    kpts = z["kpts"]
    desc = z["desc"]
    scores = z["scores"]
    gdesc = z["global_desc"]
    has_g = gdesc.ndim == 2 and len(gdesc) == len(stamps)
    blank = np.zeros((1, 1, 3), dtype=np.uint8)
    frames: List[FrameData] = []
    for i in range(len(stamps)):
        g = tuple(float(v) for v in xyzyaw[i])
        frames.append(
            FrameData(
                stamp_ns=int(stamps[i]),
                bgr_undist=blank,
                T_map_cam=np.asarray(Tmc[i], dtype=np.float64),
                gt_xyzyaw=(g[0], g[1], g[2], g[3]),
                kpts=np.asarray(kpts[i]),
                desc=np.asarray(desc[i]),
                scores=np.asarray(scores[i]),
                global_desc=np.asarray(gdesc[i], dtype=np.float32) if has_g else None,
            )
        )
    return frames, points
