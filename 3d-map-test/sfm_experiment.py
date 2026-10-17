"""Helpers for Lap2 GT-posed SfM mapping + Lap3 NetVLAD/LightGlue localization."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from tqdm.auto import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
CAMLOC = ROOT.parent / "c0_2026-10-13_22-53-42_camloc"
EXTRACTED = CAMLOC / "extracted"
DEFAULT_MCAP = CAMLOC / "c0_2026-10-13_22-53-42_camloc_0.mcap"


def pick_device(device: Optional[str] = None) -> str:
    """Prefer cuda, then Apple MPS, else cpu."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def T_from_tf(tf: dict, key: str) -> np.ndarray:
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


@dataclass
class CamCalib:
    K: np.ndarray
    D: np.ndarray
    T_base_cam: np.ndarray


def load_camera_calib(camera: str = "r", extracted: Path = EXTRACTED) -> CamCalib:
    """Load K, D, T_base_cam for camera in {r, fl, fr, rl, ...}."""
    cam = camera.strip().lower()
    ci = json.loads((extracted / "camera_info.json").read_text())
    tf = json.loads((extracted / "tf_static.json").read_text())
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


def load_camera_r_calib(extracted: Path = EXTRACTED) -> CamCalib:
    return load_camera_calib("r", extracted)


def load_gt(extracted: Path = EXTRACTED) -> pd.DataFrame:
    gt = pd.read_csv(extracted / "gt.csv", dtype={"timestamp_ns": str})
    gt["timestamp_ns"] = gt["timestamp_ns"].astype("int64")
    return gt


def load_lap(lap: int, laps_json: Path = ROOT / "lapdata" / "laps.json") -> dict:
    with open(laps_json) as f:
        return next(x for x in json.load(f)["laps"] if x["lap"] == lap)


def gt_at(gt: pd.DataFrame, ts: int) -> Tuple[float, float, float, float]:
    i = int(np.argmin(np.abs(gt["timestamp_ns"].to_numpy(np.int64) - np.int64(ts))))
    r = gt.iloc[i]
    return float(r.x), float(r.y), float(r.z), float(r.yaw)


def undistort(bgr: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    return cv2.undistort(bgr, K, D, None, K)


def T_map_cam(gt_pose, T_base_cam: np.ndarray) -> np.ndarray:
    return T_map_base(*gt_pose) @ T_base_cam


def relative_T(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """Maps points from frame b to frame a: X_a = T_a←b X_b."""
    return np.linalg.inv(T_a) @ T_b


def skew(t: np.ndarray) -> np.ndarray:
    x, y, z = t.ravel()
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def essential_from_T(T_a_b: np.ndarray) -> np.ndarray:
    """E such that x_a^T E x_b = 0 for normalized coords; X_a = R X_b + t."""
    R, t = T_a_b[:3, :3], T_a_b[:3, 3]
    return skew(t) @ R


def epi_distances(F: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    ph0 = np.c_[p0, np.ones(len(p0))].T
    ph1 = np.c_[p1, np.ones(len(p1))].T
    l1 = F @ ph0
    return np.abs(np.sum(ph1 * l1, axis=0)) / np.maximum(
        np.sqrt(l1[0] ** 2 + l1[1] ** 2), 1e-12
    )


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


class LocalFeatures:
    def __init__(self, device: Optional[str] = None, max_keypoints: int = 2048, resize: int = 512):
        self.device = pick_device(device)
        self.resize = resize
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    @torch.no_grad()
    def extract(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats = self.extractor.extract(
            numpy_image_to_torch(bgr).to(self.device), resize=self.resize
        )
        feats = rbd(feats)
        kpts = feats["keypoints"].detach().cpu().numpy()
        desc = feats["descriptors"].detach().cpu().numpy()
        scores = feats["keypoint_scores"].detach().cpu().numpy()
        return kpts, desc, scores

    @torch.no_grad()
    def match(
        self, k0: np.ndarray, d0: np.ndarray, k1: np.ndarray, d1: np.ndarray, size0, size1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return matched pixel coords (p0, p1). size = (W, H)."""
        if len(k0) == 0 or len(k1) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        f0 = {
            "keypoints": torch.from_numpy(np.asarray(k0, np.float32))[None].to(self.device),
            "descriptors": torch.from_numpy(np.asarray(d0, np.float32))[None].to(self.device),
            "image_size": torch.tensor([[size0[0], size0[1]]], dtype=torch.float32).to(
                self.device
            ),
        }
        f1 = {
            "keypoints": torch.from_numpy(np.asarray(k1, np.float32))[None].to(self.device),
            "descriptors": torch.from_numpy(np.asarray(d1, np.float32))[None].to(self.device),
            "image_size": torch.tensor([[size1[0], size1[1]]], dtype=torch.float32).to(
                self.device
            ),
        }
        m = rbd(self.matcher({"image0": f0, "image1": f1}))
        mt = m["matches"].detach().cpu().numpy()
        if len(mt) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        return k0[mt[:, 0]], k1[mt[:, 1]]

def filter_ambiguous_descriptors(
    kpts: np.ndarray,
    desc: np.ndarray,
    scores: np.ndarray,
    *,
    min_nn_dist: float = 0.5,
    max_keep: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop keypoints whose descriptor is too similar to another in the same image.

    SuperPoint descriptors are L2-normalized; cosine distance = 1 - dot.
    We keep points with nearest-neighbor L2 distance >= min_nn_dist.
    """
    n = len(kpts)
    if n < 2:
        return kpts, desc, scores
    # desc: (N, D) after transpose if needed
    D = desc if desc.ndim == 2 and desc.shape[0] == n else desc.T
    # pairwise dots
    sim = D @ D.T
    np.fill_diagonal(sim, -1.0)
    nn_sim = sim.max(axis=1)
    nn_dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * nn_sim))  # L2 for unit vectors
    keep = nn_dist >= min_nn_dist
    kpts, D, scores = kpts[keep], D[keep], scores[keep]
    if max_keep is not None and len(kpts) > max_keep:
        order = np.argsort(-scores)[:max_keep]
        kpts, D, scores = kpts[order], D[order], scores[order]
    return kpts, D, scores


class NetVLADGlobal:
    """Thin wrapper around hloc NetVLAD (downloads Pitts30K weights on first use)."""

    def __init__(self, device: Optional[str] = None):
        import sys

        hloc_root = ROOT / "third_party" / "Hierarchical-Localization"
        if str(hloc_root) not in sys.path:
            sys.path.insert(0, str(hloc_root))
        from hloc.extractors.netvlad import NetVLAD

        self.device = pick_device(device)
        self.model = NetVLAD({"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}).eval()
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def extract(self, bgr: np.ndarray) -> np.ndarray:
        # hloc NetVLAD: RGB float in [0, 1]; internal path scales to 0–255
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        h, w = img.shape[-2:]
        scale = 1024 / max(h, w)
        if scale < 1:
            img = torch.nn.functional.interpolate(
                img[None], scale_factor=scale, mode="bilinear", align_corners=False
            )[0]
        out = self.model({"image": img[None].to(self.device)})
        g = out["global_descriptor"][0].detach().cpu().numpy().astype(np.float32)
        g = g / (np.linalg.norm(g) + 1e-12)
        return g


def extract_frames_from_mcap(
    mcap: Path,
    start_ns: int,
    end_ns: int,
    calib: CamCalib,
    gt: pd.DataFrame,
    *,
    camera: str = "r",
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> List[FrameData]:
    from camloc_play import iter_camera_frames

    cam = camera.strip().lower()
    frames: List[FrameData] = []
    # Rough upper bound for progress (10 Hz); stops early on max_frames.
    est = None
    if max_frames is not None:
        est = int(max_frames) * int(stride)
    elif end_ns is not None and start_ns is not None:
        est = max(1, int((end_ns - start_ns) / 1e8))  # ~10 Hz
    it = iter_camera_frames(
        mcap, cameras=(cam,), start_ns=start_ns, end_ns=end_ns, max_frames=None
    )
    pbar = tqdm(it, total=est, desc=f"load camera_{cam}", unit="frm")
    for i, fr in enumerate(pbar):
        if i % stride != 0:
            continue
        gpose = gt_at(gt, fr.stamp_ns)
        und = undistort(fr.bgr, calib.K, calib.D)
        frames.append(
            FrameData(
                stamp_ns=fr.stamp_ns,
                bgr_undist=und,
                T_map_cam=T_map_cam(gpose, calib.T_base_cam),
                gt_xyzyaw=gpose,
            )
        )
        pbar.set_postfix(kept=len(frames))
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


def run_local_and_global(
    frames: List[FrameData],
    local: LocalFeatures,
    netvlad: Optional[NetVLADGlobal],
    *,
    min_nn_dist: float = 0.5,
) -> None:
    for fr in tqdm(frames, desc="SuperPoint(+NetVLAD)", unit="frm"):
        k, d, s = local.extract(fr.bgr_undist)
        k, d, s = filter_ambiguous_descriptors(k, d, s, min_nn_dist=min_nn_dist)
        fr.kpts, fr.desc, fr.scores = k, d, s
        if netvlad is not None:
            fr.global_desc = netvlad.extract(fr.bgr_undist)


@dataclass
class MapPoint:
    xyz: np.ndarray
    observations: List[Tuple[int, int]] = field(default_factory=list)  # (frame_idx, kpt_idx) optional
    color: Optional[np.ndarray] = None


def triangulate_pair(
    K: np.ndarray,
    T_map_a: np.ndarray,
    T_map_b: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    *,
    epi_thresh_px: float = 6.0,
    z_range: Tuple[float, float] = (2.0, 80.0),
    reproj_thresh: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate in map frame. Returns (X_map Nx3, mask)."""
    if len(p_a) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=bool)
    T_a_b = relative_T(T_map_a, T_map_b)  # X_a = R X_b + t
    E = essential_from_T(T_a_b)
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    d = epi_distances(F, p_a, p_b)
    inl = d < epi_thresh_px
    if inl.sum() < 8:
        return np.zeros((0, 3)), inl

    P_a = K @ np.linalg.inv(T_map_a)[:3, :]
    P_b = K @ np.linalg.inv(T_map_b)[:3, :]
    Xh = cv2.triangulatePoints(P_a, P_b, p_a[inl].T, p_b[inl].T)
    X = (Xh[:3] / Xh[3:]).T

    # cheirality in both cameras
    Xa = (np.linalg.inv(T_map_a) @ np.c_[X, np.ones(len(X))].T).T[:, :3]
    Xb = (np.linalg.inv(T_map_b) @ np.c_[X, np.ones(len(X))].T).T[:, :3]
    ok = (
        (Xa[:, 2] > z_range[0])
        & (Xb[:, 2] > z_range[0])
        & (Xa[:, 2] < z_range[1])
        & (Xb[:, 2] < z_range[1])
        & np.isfinite(X).all(1)
    )
    # reprojection
    for P, p, Xc in ((P_a, p_a[inl], X), (P_b, p_b[inl], X)):
        proj = (P @ np.c_[Xc, np.ones(len(Xc))].T).T
        proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)
        ok &= np.linalg.norm(proj - p, axis=1) < reproj_thresh

    mask_full = np.zeros(len(p_a), dtype=bool)
    idx = np.where(inl)[0][ok]
    mask_full[idx] = True
    return X[ok], mask_full


def build_map(
    frames: List[FrameData],
    local: LocalFeatures,
    K: np.ndarray,
    *,
    window: int = 2,
    min_baseline_m: float = 0.8,
    epi_thresh_px: float = 6.0,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Return (points Nx3, observations list)."""
    points: List[np.ndarray] = []
    observations: List[Dict[str, Any]] = []
    h, w = frames[0].bgr_undist.shape[:2]
    size = (w, h)

    for i in tqdm(range(len(frames)), desc="SfM triangulate", unit="frm"):
        for j in range(i + 1, min(i + window + 1, len(frames))):
            Ta, Tb = frames[i].T_map_cam, frames[j].T_map_cam
            baseline = np.linalg.norm(relative_T(Ta, Tb)[:3, 3])
            if baseline < min_baseline_m:
                continue
            p0, p1 = local.match(
                frames[i].kpts,
                frames[i].desc,
                frames[j].kpts,
                frames[j].desc,
                size,
                size,
            )
            X, mask = triangulate_pair(
                K, Ta, Tb, p0, p1, epi_thresh_px=epi_thresh_px
            )
            for x in X:
                pid = len(points)
                points.append(x)
                observations.append(
                    {"point_id": pid, "frame_i": i, "frame_j": j, "xyz": x}
                )

    if not points:
        return np.zeros((0, 3)), observations
    return np.stack(points, 0), observations


def project_points(K: np.ndarray, T_map_cam: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    local: LocalFeatures,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    *,
    top_k: int = 5,
    pnp_reproj: float = 4.0,
    min_inl: int = 12,
) -> Dict[str, Any]:
    db_g = np.stack([f.global_desc for f in map_frames], 0)
    idx, sims = retrieve_topk(query.global_desc, db_g, k=top_k)
    h, w = query.bgr_undist.shape[:2]
    size = (w, h)

    obj_list, img_list = [], []
    cand_stats = []
    for ci in idx:
        c = map_frames[int(ci)]
        p_q, p_c = local.match(query.kpts, query.desc, c.kpts, c.desc, size, size)
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
    # OpenCV: X_cam = R X_world + t  with world = map
    T_cam_map = np.eye(4)
    T_cam_map[:3, :3] = Rm
    T_cam_map[:3, 3] = tvec.ravel()
    T_map_cam = np.linalg.inv(T_cam_map)
    T_map_base = T_map_cam @ np.linalg.inv(T_base_cam)
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


def save_ply(path: Path, pts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


MAP_BUNDLE_NAME = "map_bundle.npz"


def map_ready(map_dir: Path) -> bool:
    """True if a reloadable map bundle exists (features + points)."""
    d = Path(map_dir)
    return (d / MAP_BUNDLE_NAME).is_file()


def save_map_bundle(
    map_dir: Path,
    frames: List[FrameData],
    points: np.ndarray,
) -> Path:
    """Persist map frames (features/poses, no images) + points for skip-rebuild."""
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("save_map_bundle: empty frames")
    gstack = None
    if frames[0].global_desc is not None:
        gstack = np.stack([f.global_desc for f in frames], 0).astype(np.float32)
    path = map_dir / MAP_BUNDLE_NAME
    np.savez_compressed(
        path,
        points=np.asarray(points, dtype=np.float64),
        stamps=np.array([f.stamp_ns for f in frames], dtype=np.int64),
        xyzyaw=np.array([f.gt_xyzyaw for f in frames], dtype=np.float64),
        T_map_cam=np.stack([np.asarray(f.T_map_cam, dtype=np.float64) for f in frames], 0),
        kpts=np.array([f.kpts for f in frames], dtype=object),
        desc=np.array([f.desc for f in frames], dtype=object),
        scores=np.array([f.scores for f in frames], dtype=object),
        global_desc=gstack if gstack is not None else np.zeros((0, 0), dtype=np.float32),
    )
    save_ply(map_dir / "points3D.ply", points)
    if gstack is not None:
        np.savez_compressed(
            map_dir / "netvlad_db.npz",
            descriptors=gstack,
            stamps=np.array([f.stamp_ns for f in frames], dtype=np.int64),
            xyzyaw=np.array([f.gt_xyzyaw for f in frames], dtype=np.float64),
        )
    return path


def load_map_bundle(map_dir: Path) -> Tuple[List[FrameData], np.ndarray]:
    """Load frames (placeholder images) + points from map_bundle.npz."""
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
