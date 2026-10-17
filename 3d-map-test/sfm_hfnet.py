"""HF-Net style hierarchical localization helpers (arXiv:1812.03506).

Paper idea:
  - one CNN → local keypoints/descriptors + global descriptor
  - retrieve top-K with global desc
  - match locals with mutual NN + ratio test (not SuperGlue/LightGlue)
  - PnP

Backends
--------
- ``hfnet_onnx``: real HF-Net via ONNX Runtime (place ``models/hfnet.onnx``).
- ``sp_nv_nn``: SuperPoint + NetVLAD + NN match — same *pipeline*, not the
  monolithic MobileNet; useful to measure NN-vs-LightGlue speedup before ONNX.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm.auto import tqdm

import sfm_experiment as sfm

ROOT = Path(__file__).resolve().parent
PIXEL_ROOT = ROOT.parent
MODELS_DIR = ROOT / "models"
DEFAULT_ONNX = MODELS_DIR / "hfnet.onnx"
DEFAULT_POLYGONS = PIXEL_ROOT / "fr-auto-mask" / "polygons.json"

# Common community mirrors / placeholders — override with HFNET_ONNX_URL env if needed.
# Official ETH dump moved; user may convert SavedModel → ONNX (see models/README_HFNET.md).
HFNET_ONNX_CANDIDATES: Tuple[str, ...] = ()


def mutual_nn_match(
    k0: np.ndarray,
    d0: np.ndarray,
    k1: np.ndarray,
    d1: np.ndarray,
    *,
    ratio: float = 0.8,
    cross_check: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mutual nearest-neighbor + Lowe ratio on L2-normalized descriptors (paper)."""
    if len(k0) == 0 or len(k1) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    D0 = d0 if d0.ndim == 2 and d0.shape[0] == len(k0) else d0.T
    D1 = d1 if d1.ndim == 2 and d1.shape[0] == len(k1) else d1.T
    D0 = D0.astype(np.float32)
    D1 = D1.astype(np.float32)
    # cosine sim for unit vectors; larger = better
    sim = D0 @ D1.T  # (N0, N1)
    # NN and 2nd NN for ratio on distance = sqrt(2-2sim)
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
    """Duck-types LocalFeatures.match for sfm.build_map / localize_query."""

    def __init__(self, ratio: float = 0.8):
        self.ratio = ratio

    def match(self, k0, d0, k1, d1, size0, size1):
        return mutual_nn_match(k0, d0, k1, d1, ratio=self.ratio)


class SPNVJointFeatures(NNMatcher):
    """Proxy backend: SuperPoint + NetVLAD, joint API, NN match (not true HF-Net)."""

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
        # expose LightGlue-less matcher only
        self.matcher = None

    def extract_joint(
        self, bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kpts, desc, scores = self.sp.extract(bgr)
        g = self.netvlad.extract(bgr)
        return kpts, desc, scores, g


class HFNetONNX(NNMatcher):
    """Real HF-Net (MobileNet joint heads) via ONNX Runtime.

    Expected outputs (name-flexible): keypoints, scores, local_descriptors, global_descriptor.
    Input: NCHW float RGB or gray — auto-probed on first call.
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
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"HF-Net ONNX not found: {self.onnx_path}\n"
                f"See {MODELS_DIR / 'README_HFNET.md'}"
            )
        self.device = sfm.pick_device(device)
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # CoreML EP sometimes available on macOS
        if self.device == "mps":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=providers
        )
        self.resize = resize
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self._input_name = self.session.get_inputs()[0].name
        self._outputs = [o.name for o in self.session.get_outputs()]

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = self.resize / max(h, w)
        if scale != 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        self._scale = scale
        # try RGB float [0,1] NCHW
        x = rgb.astype(np.float32) / 255.0
        return x.transpose(2, 0, 1)[None]

    def extract_joint(
        self, bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        inp = self._preprocess(bgr)
        outs = self.session.run(self._outputs, {self._input_name: inp})
        by_name = dict(zip(self._outputs, outs))

        def pick(*keys):
            for k in keys:
                for name, val in by_name.items():
                    if k in name.lower():
                        return val
            return None

        dense_desc = pick("local_descriptor", "descriptors", "desc")
        scores_map = pick("keypoint", "score", "prob")
        global_desc = pick("global", "netvlad")
        if global_desc is None:
            raise RuntimeError(f"HF-Net ONNX missing global head; outs={self._outputs}")

        g = np.asarray(global_desc).reshape(-1).astype(np.float32)
        g = g / (np.linalg.norm(g) + 1e-12)

        # If model already returns sparse kpts
        kpts_out = pick("keypoints")
        if kpts_out is not None and np.asarray(kpts_out).ndim >= 2 and np.asarray(kpts_out).shape[-1] == 2:
            kpts = np.asarray(kpts_out).reshape(-1, 2).astype(np.float32)
            if "scores" in "".join(by_name):
                scores = np.asarray(pick("scores")).reshape(-1).astype(np.float32)
            else:
                scores = np.ones(len(kpts), np.float32)
            desc = np.asarray(dense_desc)
            if desc.ndim == 3:
                # sample later — fall through
                kpts_out = None
            else:
                if desc.ndim == 2 and desc.shape[0] != len(kpts) and desc.shape[1] == len(kpts):
                    desc = desc.T
                kpts = kpts / max(self._scale, 1e-9)
                return kpts, desc.astype(np.float32), scores, g

        # Dense SuperPoint-style decode
        if scores_map is None or dense_desc is None:
            raise RuntimeError(
                f"Cannot decode sparse features from ONNX outs={self._outputs}"
            )
        sm = np.asarray(scores_map)
        if sm.ndim == 4:
            sm = sm[0, 0] if sm.shape[1] == 1 else sm[0].max(0)
        elif sm.ndim == 3:
            sm = sm[0]
        dd = np.asarray(dense_desc)
        if dd.ndim == 4:
            dd = dd[0]  # C,H,W
        # NMS-ish: take top peaks
        flat = sm.reshape(-1)
        thr = self.keypoint_threshold
        cand = np.where(flat >= thr)[0]
        if len(cand) == 0:
            cand = np.argsort(-flat)[: self.max_keypoints]
        else:
            order = np.argsort(-flat[cand])[: self.max_keypoints]
            cand = cand[order]
        ys, xs = np.unravel_index(cand, sm.shape[-2:])
        # descriptor cell often 8× coarser
        cell = max(1, int(round(bgr.shape[0] * self._scale / dd.shape[-2])))
        kpts = np.stack([xs * cell, ys * cell], 1).astype(np.float32)
        # sample desc
        cy = np.clip((ys).astype(int), 0, dd.shape[-2] - 1)
        cx = np.clip((xs).astype(int), 0, dd.shape[-1] - 1)
        desc = dd[:, cy, cx].T.astype(np.float32)
        desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12
        scores = flat[cand].astype(np.float32)
        kpts = kpts / max(self._scale, 1e-9)
        return kpts, desc, scores, g


def make_features(
    backend: str = "sp_nv_nn",
    **kwargs,
) -> Union[SPNVJointFeatures, HFNetONNX]:
    backend = backend.lower().strip()
    if backend in ("hfnet", "hfnet_onnx", "onnx"):
        return HFNetONNX(**kwargs)
    if backend in ("sp_nv_nn", "proxy", "sp+nv"):
        return SPNVJointFeatures(**kwargs)
    raise ValueError(f"unknown backend {backend!r}; use hfnet_onnx or sp_nv_nn")


def load_ego_mask(
    camera: str,
    polygons_path: Path = DEFAULT_POLYGONS,
    *,
    out_hw: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """Build uint8 mask: 255=feature allowed, 0=ego / ignored (from polygons.json).

    ``masked_polygons[cam]`` are filled with 0. Returns None if no polygons.
    """
    cam = camera.strip().lower()
    data = json.loads(Path(polygons_path).read_text())
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
    """Zero out ignored pixels (mask==0)."""
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
    frames: List[sfm.FrameData],
    feats: Union[SPNVJointFeatures, HFNetONNX],
    *,
    min_nn_dist: float = 0.5,
    ego_mask: Optional[np.ndarray] = None,
    desc: str = "HFNet-joint",
) -> None:
    for fr in tqdm(frames, desc=desc, unit="frm"):
        img = apply_ego_mask_bgr(fr.bgr_undist, ego_mask)
        kpts, d, scores, g = feats.extract_joint(img)
        kpts, d, scores = filter_kpts_by_mask(kpts, d, scores, ego_mask)
        kpts, d, scores = sfm.filter_ambiguous_descriptors(
            kpts, d, scores, min_nn_dist=min_nn_dist
        )
        fr.kpts, fr.desc, fr.scores = kpts, d, scores
        fr.global_desc = g


def run_camera_pipeline(
    camera: str,
    *,
    feats: Union[SPNVJointFeatures, HFNetONNX],
    calib: sfm.CamCalib,
    gt,
    lap_map: dict,
    lap_q: dict,
    out_root: Path,
    map_max_frames: Optional[int],
    query_max_frames: Optional[int],
    map_stride: int,
    query_stride: int,
    min_nn_dist: float,
    match_window: int,
    epi_thresh_px: float,
    top_k: int,
    override_map: bool,
    polygons_path: Path = DEFAULT_POLYGONS,
) -> Dict:
    """Independent Lap2 map + Lap3 query for one camera. Returns summary dict."""
    cam = camera.strip().lower()
    map_dir = Path(out_root) / f"map_{cam}"
    loc_dir = Path(out_root) / f"loc_lap3_{cam}"
    map_dir.mkdir(parents=True, exist_ok=True)
    loc_dir.mkdir(parents=True, exist_ok=True)

    ego_mask = load_ego_mask(cam, polygons_path)
    if ego_mask is not None:
        print(f"[{cam}] ego mask from {polygons_path}  ignored_px={(ego_mask == 0).mean()*100:.1f}%")
    else:
        print(f"[{cam}] no ego polygons — full frame features")

    skip = sfm.map_ready(map_dir) and not override_map

    if skip:
        print(f"[{cam}] LOAD map", map_dir / sfm.MAP_BUNDLE_NAME)
        map_frames, points = sfm.load_map_bundle(map_dir)
    else:
        t0 = time.perf_counter()
        map_frames = sfm.extract_frames_from_mcap(
            sfm.DEFAULT_MCAP,
            int(lap_map["start_ns"]),
            int(lap_map["end_ns"]),
            calib,
            gt,
            camera=cam,
            max_frames=map_max_frames,
            stride=map_stride,
        )
        print(f"[{cam}] map extract {len(map_frames)} in {time.perf_counter()-t0:.1f}s")
        t0 = time.perf_counter()
        run_joint_features(
            map_frames,
            feats,
            min_nn_dist=min_nn_dist,
            ego_mask=ego_mask,
            desc=f"feat-map-{cam}",
        )
        print(f"[{cam}] map features {time.perf_counter()-t0:.1f}s")
        t0 = time.perf_counter()
        points, _obs = sfm.build_map(
            map_frames,
            feats,
            calib.K,
            window=match_window,
            min_baseline_m=0.8,
            epi_thresh_px=epi_thresh_px,
        )
        print(f"[{cam}] points={len(points)} in {time.perf_counter()-t0:.1f}s")
        sfm.save_map_bundle(map_dir, map_frames, points)

    t0 = time.perf_counter()
    query_frames = sfm.extract_frames_from_mcap(
        sfm.DEFAULT_MCAP,
        int(lap_q["start_ns"]),
        int(lap_q["end_ns"]),
        calib,
        gt,
        camera=cam,
        max_frames=query_max_frames,
        stride=query_stride,
    )
    print(f"[{cam}] query extract {len(query_frames)} in {time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter()
    run_joint_features(
        query_frames,
        feats,
        min_nn_dist=min_nn_dist,
        ego_mask=ego_mask,
        desc=f"feat-qry-{cam}",
    )
    print(f"[{cam}] query features {time.perf_counter()-t0:.1f}s")

    results = []
    loc_times = []
    t0 = time.perf_counter()
    for i, q in enumerate(tqdm(query_frames, desc=f"loc-{cam}", unit="qry")):
        t_i = time.perf_counter()
        r = sfm.localize_query(
            q, map_frames, points, feats, calib.K, calib.T_base_cam, top_k=top_k
        )
        loc_times.append(time.perf_counter() - t_i)
        results.append(r)
        if i < 10 or (i + 1) % 50 == 0:
            tqdm.write(
                f"[{cam} {i}] ok={r['ok']} corr={r['n_pnp_corr']} "
                f"inl={r['n_pnp_inl']} err={r['err_xy']}"
            )
    loc_total = time.perf_counter() - t0
    print(
        f"[{cam}] localize {loc_total:.1f}s  "
        f"({1000*np.mean(loc_times):.1f} ms/frame mean)"
    )

    return {
        "camera": cam,
        "map_dir": map_dir,
        "loc_dir": loc_dir,
        "map_frames": map_frames,
        "query_frames": query_frames,
        "points": points,
        "results": results,
        "ego_mask": ego_mask,
        "skipped_map": skip,
        "localize_times_s": np.asarray(loc_times, dtype=np.float64),
        "localize_total_s": float(loc_total),
    }


def _rms(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v**2))) if len(v) else float("nan")


def _wrap_deg(a: np.ndarray) -> np.ndarray:
    return np.degrees(np.angle(np.exp(1j * np.asarray(a, dtype=np.float64))))


DEFAULT_MAP_MCAP = (
    PIXEL_ROOT.parent
    / "c0_2026-10-13_22-53-42_camloc"
    / "23-00-38_cut_120m_update"
    / "map.mcap"
)


def load_racing_line(map_mcap: Path = DEFAULT_MAP_MCAP):
    """Load racing/pit lines from map.mcap (option B / track_frame)."""
    tools = Path(map_mcap).resolve().parent / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from track_frame import load_lines  # type: ignore

    return load_lines(str(map_mcap))


def compute_camloc_metrics(
    *,
    stamps_ns: np.ndarray,
    gt_xy: np.ndarray,
    gt_yaw: np.ndarray,
    pred_xy: np.ndarray,
    pred_yaw: np.ndarray,
    ok: np.ndarray,
    localize_times_s: Optional[np.ndarray] = None,
    map_mcap: Path = DEFAULT_MAP_MCAP,
) -> Dict:
    """Official-style metrics (camloc README + cut_120m evaluate.py).

    Option A: map x/y (+ car-frame lat/lon) and yaw.
    Option B: racing-line (s, n) via ``track_frame.py`` + map.mcap.
    """
    ok = np.asarray(ok, dtype=bool)
    n_all = int(len(ok))
    n_ok = int(ok.sum())
    out: Dict = {
        "n_query": n_all,
        "n_success": n_ok,
        "success_rate_%": 100.0 * n_ok / max(n_all, 1),
    }

    if localize_times_s is not None and len(localize_times_s):
        t = np.asarray(localize_times_s, dtype=np.float64)
        out["runtime_ms_mean"] = float(np.mean(t) * 1000)
        out["runtime_ms_median"] = float(np.median(t) * 1000)
        out["runtime_ms_p95"] = float(np.percentile(t, 95) * 1000)
    else:
        out["runtime_ms_mean"] = out["runtime_ms_median"] = out["runtime_ms_p95"] = float(
            "nan"
        )

    if n_ok == 0:
        for k in (
            "pos_rmse_xy_m",
            "lat_rmse_m",
            "lon_rmse_m",
            "yaw_rmse_deg",
            "within_1m_%",
            "within_1m_of_all_%",
            "pos_rmse_sn_m",
            "s_rmse_m",
            "n_rmse_m",
        ):
            out[k] = float("nan")
        return out

    gx, gy = gt_xy[ok, 0], gt_xy[ok, 1]
    px, py = pred_xy[ok, 0], pred_xy[ok, 1]
    gyaw, pyaw = gt_yaw[ok], pred_yaw[ok]
    dx, dy = px - gx, py - gy
    pos = np.hypot(dx, dy)
    lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
    lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
    yaw_err = _wrap_deg(pyaw - gyaw)

    out["pos_rmse_xy_m"] = _rms(pos)  # Option A score
    out["lat_rmse_m"] = _rms(lat)
    out["lon_rmse_m"] = _rms(lon)
    out["yaw_rmse_deg"] = _rms(yaw_err)
    out["within_1m_%"] = 100.0 * float(np.mean(pos < 1.0))  # among successful
    out["within_1m_of_all_%"] = 100.0 * float(np.sum(pos < 1.0) / max(n_all, 1))
    out["pos_median_m"] = float(np.median(pos))
    out["pos_p95_m"] = float(np.percentile(pos, 95))

    # Option B: s/n on racing line
    out["pos_rmse_sn_m"] = float("nan")
    out["s_rmse_m"] = float("nan")
    out["n_rmse_m"] = float("nan")
    map_mcap = Path(map_mcap)
    if map_mcap.is_file():
        lines = load_racing_line(map_mcap)
        lap = lines["track"].lap
        ts = np.asarray(stamps_ns, dtype=np.int64)[ok]
        sg, ng, _, _ = lines["track"].xy_to_sn(gx, gy, gyaw, ts)
        sp, np_, _, _ = lines["track"].xy_to_sn(px, py, pyaw, ts)
        # unwrap GT s then wrap ds
        sg_u = np.unwrap(sg, period=lap)
        ds = (sp - sg_u + lap / 2) % lap - lap / 2
        dn = np_ - ng
        out["s_rmse_m"] = _rms(ds)
        out["n_rmse_m"] = _rms(dn)
        out["pos_rmse_sn_m"] = _rms(np.hypot(ds, dn))
        out["track_lap_m"] = float(lap)
    else:
        out["option_b_error"] = f"map.mcap missing: {map_mcap}"

    return out


def pose_confidence(
    ok: bool,
    n_inl: int,
    *,
    e_reproj: Optional[float] = None,
    sigma_reproj: float = 4.0,
) -> float:
    """ω = 1[ok] · n_inl · exp(-ē_reproj / σ)  (SFM_FEASIBILITY §10.5)."""
    if not ok or n_inl <= 0:
        return 0.0
    w = float(n_inl)
    if e_reproj is not None and np.isfinite(e_reproj):
        w *= float(np.exp(-float(e_reproj) / max(sigma_reproj, 1e-6)))
    return w


def _wrap_pi(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def fuse_se2_weighted(
    poses: List[Tuple[float, float, float]],
    weights: List[float],
    *,
    gate_xy_m: float = 5.0,
    gate_yaw_deg: float = 15.0,
) -> Tuple[Optional[Tuple[float, float, float]], Dict]:
    """Fuse base_link (x,y,yaw) with confidence weights + disagreement gating.

    Returns (fused_pose or None, info).
    """
    items = [(p, float(w)) for p, w in zip(poses, weights) if w > 0 and p is not None]
    info: Dict = {"n_in": len(items), "mode": None, "weights": []}
    if not items:
        return None, info
    items.sort(key=lambda pw: -pw[1])
    (x0, y0, yaw0), w0 = items[0]
    kept = [(x0, y0, yaw0, w0)]
    for (x, y, yaw), w in items[1:]:
        dxy = float(np.hypot(x - x0, y - y0))
        dyaw = abs(_wrap_pi(yaw - yaw0)) * 180 / np.pi
        if dxy <= gate_xy_m and dyaw <= gate_yaw_deg:
            kept.append((x, y, yaw, w))
        # else drop conflicting lower-weight pose
    info["n_kept"] = len(kept)
    info["weights"] = [k[3] for k in kept]
    if len(kept) == 1:
        info["mode"] = "single"
        return (kept[0][0], kept[0][1], kept[0][2]), info
    ws = np.array([k[3] for k in kept], dtype=np.float64)
    ws = ws / ws.sum()
    x = float(np.sum(ws * np.array([k[0] for k in kept])))
    y = float(np.sum(ws * np.array([k[1] for k in kept])))
    sy = float(np.sum(ws * np.sin([k[2] for k in kept])))
    cy = float(np.sum(ws * np.cos([k[2] for k in kept])))
    yaw = float(np.arctan2(sy, cy))
    info["mode"] = "se2_weighted"
    return (x, y, yaw), info


def fuse_hard_select(
    poses: List[Tuple[float, float, float]],
    weights: List[float],
) -> Tuple[Optional[Tuple[float, float, float]], Dict]:
    """Pick the pose with max ω (hard select)."""
    best_i, best_w = -1, 0.0
    for i, w in enumerate(weights):
        if w > best_w and poses[i] is not None:
            best_i, best_w = i, w
    if best_i < 0:
        return None, {"mode": None, "n_in": 0}
    return poses[best_i], {"mode": "hard_select", "n_in": 1, "weights": [best_w], "index": best_i}


def align_by_nearest_stamp(
    ref_stamps: np.ndarray,
    other_stamps: np.ndarray,
) -> np.ndarray:
    """For each ref stamp, index of nearest other stamp."""
    ref_stamps = np.asarray(ref_stamps, dtype=np.int64)
    other_stamps = np.asarray(other_stamps, dtype=np.int64)
    idx = np.searchsorted(other_stamps, ref_stamps)
    out = np.empty(len(ref_stamps), dtype=np.int64)
    for i, (t, j) in enumerate(zip(ref_stamps, idx)):
        cands = [k for k in (int(j), int(j) - 1) if 0 <= k < len(other_stamps)]
        out[i] = min(cands, key=lambda k: abs(int(other_stamps[k]) - int(t)))
    return out


def fuse_multi_camera_tables(
    tables: Dict[str, "pd.DataFrame"],
    *,
    ref_camera: str = "r",
    mode: str = "se2",  # "se2" | "hard"
    gate_xy_m: float = 5.0,
    gate_yaw_deg: float = 15.0,
    sigma_reproj: float = 4.0,
) -> "pd.DataFrame":
    """Time-align camera error tables and fuse base_link poses (no feature fusion).

    Each table needs: stamp_ns, ok, gt_x, gt_y, gt_yaw, pred_x, pred_y, pred_yaw, n_pnp_inl.
    Optional: e_reproj for the exp term in ω.
    """
    import pandas as pd

    if ref_camera not in tables:
        ref_camera = next(iter(tables))
    ref = tables[ref_camera].sort_values("stamp_ns").reset_index(drop=True)
    cams = list(tables.keys())
    aligned = {ref_camera: ref}
    for c in cams:
        if c == ref_camera:
            continue
        df = tables[c].sort_values("stamp_ns").reset_index(drop=True)
        ji = align_by_nearest_stamp(ref["stamp_ns"].to_numpy(), df["stamp_ns"].to_numpy())
        aligned[c] = df.iloc[ji].reset_index(drop=True)

    rows = []
    for i in range(len(ref)):
        poses = []
        weights = []
        src_ok = {}
        for c in cams:
            row = aligned[c].iloc[i]
            ok = bool(row.ok)
            src_ok[c] = ok
            if ok and pd.notna(row.pred_x):
                poses.append((float(row.pred_x), float(row.pred_y), float(row.pred_yaw)))
                e_rep = float(row.e_reproj) if "e_reproj" in row.index and pd.notna(row.e_reproj) else None
                weights.append(
                    pose_confidence(ok, int(row.n_pnp_inl), e_reproj=e_rep, sigma_reproj=sigma_reproj)
                )
            else:
                poses.append(None)
                weights.append(0.0)

        if mode == "hard":
            fused, info = fuse_hard_select(poses, weights)
        else:
            fused, info = fuse_se2_weighted(
                poses, weights, gate_xy_m=gate_xy_m, gate_yaw_deg=gate_yaw_deg
            )

        gx, gy, gyaw = float(ref.iloc[i].gt_x), float(ref.iloc[i].gt_y), float(ref.iloc[i].gt_yaw)
        ok_f = fused is not None
        if ok_f:
            px, py, pyaw = fused
            err_xy = float(np.hypot(px - gx, py - gy))
            err_yaw = abs(_wrap_pi(pyaw - gyaw)) * 180 / np.pi
        else:
            px = py = pyaw = err_xy = err_yaw = float("nan")

        rows.append(
            {
                "idx": i,
                "stamp_ns": int(ref.iloc[i].stamp_ns),
                "ok": ok_f,
                "gt_x": gx,
                "gt_y": gy,
                "gt_yaw": gyaw,
                "pred_x": px,
                "pred_y": py,
                "pred_yaw": pyaw,
                "err_xy": err_xy,
                "err_yaw_deg": err_yaw,
                "n_pnp_inl": int(max(weights)) if weights else 0,  # proxy for metrics table
                "fuse_mode": info.get("mode"),
                "n_sources": int(info.get("n_kept", info.get("n_in", 0)) or 0),
                "union_ok": bool(any(src_ok.values())),
                **{f"ok_{c}": src_ok[c] for c in cams},
                **{f"w_{c}": weights[j] for j, c in enumerate(cams)},
            }
        )
    return pd.DataFrame(rows)


def try_download_onnx(dest: Path = DEFAULT_ONNX) -> Optional[Path]:
    """Attempt download from configured candidate URLs (usually empty)."""
    dest = Path(dest)
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in HFNET_ONNX_CANDIDATES:
        try:
            print("downloading", url)
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size > 1_000_000:
                return dest
        except Exception as e:
            print("download failed:", e)
    return None


# ---------------------------------------------------------------------------
# Multi-lap map: racing-line s-order neighbors (see SFM_MULTI_LAP_S_ORDER.md)
# ---------------------------------------------------------------------------


def require_mps() -> str:
    """Hard-require Apple MPS (no silent CPU fallback)."""
    import torch

    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError(
            "MPS unavailable — multi-lap HF-Net experiment requires Apple MPS "
            "(torch.backends.mps.is_available() is False)."
        )
    return "mps"


def resolve_hfnet_backend(preferred: Optional[str] = None) -> Tuple[str, bool]:
    """Return (backend, is_proxy). Prefer hfnet_onnx when weights exist."""
    if preferred:
        b = preferred.lower().strip()
        if b in ("hfnet", "hfnet_onnx", "onnx"):
            if not DEFAULT_ONNX.is_file():
                raise FileNotFoundError(f"hfnet_onnx requested but missing: {DEFAULT_ONNX}")
            return "hfnet_onnx", False
        if b in ("sp_nv_nn", "proxy", "sp+nv"):
            return "sp_nv_nn", True
        raise ValueError(f"unknown backend {preferred!r}")
    if DEFAULT_ONNX.is_file():
        return "hfnet_onnx", False
    return "sp_nv_nn", True


def arc_distance_s(s_i: float, s_j: float, lap: float) -> float:
    d = abs(float(s_i) - float(s_j))
    return float(min(d, lap - d))


def neighbor_pairs_index(n: int, max_offset: int = 3) -> List[Tuple[int, int]]:
    """Undirected pairs for circular index neighbors i±1..±max_offset."""
    seen = set()
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for d in range(1, max_offset + 1):
            j = (i + d) % n
            key = frozenset((i, j))
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            pairs.append((i, j) if i < j else (j, i))
    return pairs


def neighbor_pairs_arc(
    s: np.ndarray,
    lap: float,
    *,
    k: int = 6,
) -> List[Tuple[int, int]]:
    """Undirected pairs: each frame → k nearest by circular arc distance on s."""
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    seen = set()
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        ds = np.abs(s - s[i])
        ds = np.minimum(ds, lap - ds)
        ds[i] = np.inf
        for j in np.argsort(ds)[:k]:
            j = int(j)
            key = frozenset((i, j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((i, j) if i < j else (j, i))
    return pairs


def project_frames_sn(
    frames: List[sfm.FrameData],
    track_line,
    *,
    n_max: Optional[float] = None,
) -> Tuple[List[sfm.FrameData], np.ndarray, np.ndarray, np.ndarray]:
    """Project GT (x,y,yaw) → (s,n,epsi), optional |n| filter, sort by s then stamp."""
    if not frames:
        z = np.zeros(0, dtype=np.float64)
        return [], z, z, z
    x = np.array([f.gt_xyzyaw[0] for f in frames], dtype=np.float64)
    y = np.array([f.gt_xyzyaw[1] for f in frames], dtype=np.float64)
    yaw = np.array([f.gt_xyzyaw[3] for f in frames], dtype=np.float64)
    ts = np.array([f.stamp_ns for f in frames], dtype=np.int64)
    s, n, epsi, _ = track_line.xy_to_sn(x, y, yaw, ts)
    keep = np.ones(len(frames), dtype=bool)
    if n_max is not None:
        keep &= np.abs(n) <= float(n_max)
    idx = np.where(keep)[0]
    order = idx[np.lexsort((ts[idx], s[idx]))]
    out_frames = [frames[int(i)] for i in order]
    return out_frames, s[order], n[order], epsi[order]


def write_frames_with_s_csv(
    path: Path,
    frames: List[sfm.FrameData],
    s: np.ndarray,
    n: np.ndarray,
    epsi: np.ndarray,
    *,
    laps: Optional[List[int]] = None,
) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stamp_ns", "lap", "s", "n", "epsi", "x", "y", "yaw"])
        for i, fr in enumerate(frames):
            lap = laps[i] if laps is not None else ""
            g = fr.gt_xyzyaw
            w.writerow(
                [
                    fr.stamp_ns,
                    lap,
                    float(s[i]),
                    float(n[i]),
                    float(epsi[i]),
                    float(g[0]),
                    float(g[1]),
                    float(g[3]),
                ]
            )
    return path


def build_map_s_neighbors(
    frames: List[sfm.FrameData],
    local,
    K: np.ndarray,
    *,
    pairs: List[Tuple[int, int]],
    min_baseline_m: float = 0.5,
    epi_thresh_px: float = 6.0,
) -> Tuple[np.ndarray, List[Dict]]:
    """Triangulate only along given undirected frame pairs (s-order neighbors)."""
    points: List[np.ndarray] = []
    observations: List[Dict] = []
    if not frames or not pairs:
        return np.zeros((0, 3)), observations
    h, w = frames[0].bgr_undist.shape[:2]
    size = (w, h)
    for i, j in tqdm(pairs, desc="SfM s-neighbors", unit="edge"):
        Ta, Tb = frames[i].T_map_cam, frames[j].T_map_cam
        baseline = np.linalg.norm(sfm.relative_T(Ta, Tb)[:3, 3])
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
        X, _mask = sfm.triangulate_pair(K, Ta, Tb, p0, p1, epi_thresh_px=epi_thresh_px)
        for x in X:
            pid = len(points)
            points.append(x)
            observations.append(
                {"point_id": pid, "frame_i": int(i), "frame_j": int(j), "xyz": x}
            )
    if not points:
        return np.zeros((0, 3)), observations
    return np.stack(points, 0), observations


def make_s_neighbor_pairs(
    s: np.ndarray,
    lap: float,
    *,
    mode: str = "arc",
    max_offset: int = 3,
    k: int = 6,
) -> List[Tuple[int, int]]:
    """mode: 'arc' (recommended) | 'index' (i±1..±max_offset)."""
    mode = mode.lower().strip()
    if mode == "arc":
        return neighbor_pairs_arc(s, lap, k=k)
    if mode == "index":
        return neighbor_pairs_index(len(s), max_offset=max_offset)
    raise ValueError(f"unknown neighbor mode {mode!r}; use 'arc' or 'index'")


QUERY_META_NAME = "query_meta.json"
LOC_TIMES_NAME = "localize_times_s.npy"


def map_bundle_fingerprint(map_dir: Path) -> Dict:
    """Identity of a saved map for query cache invalidation."""
    p = Path(map_dir) / sfm.MAP_BUNDLE_NAME
    if not p.is_file():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def loc_ready(loc_dir: Path) -> bool:
    d = Path(loc_dir)
    return (d / "errors.csv").is_file() and (d / QUERY_META_NAME).is_file()


def query_cache_valid(
    loc_dir: Path,
    *,
    map_dir: Path,
    query_lap: int,
    query_stride: int,
    top_k: int,
) -> bool:
    """True if cached query matches this map + query lap settings."""
    if not loc_ready(loc_dir):
        return False
    try:
        meta = json.loads((Path(loc_dir) / QUERY_META_NAME).read_text())
    except Exception:
        return False
    fp = map_bundle_fingerprint(map_dir)
    if not fp.get("exists"):
        return False
    cached = meta.get("map_fingerprint") or {}
    if cached.get("mtime_ns") != fp.get("mtime_ns") or cached.get("size") != fp.get(
        "size"
    ):
        return False
    if int(meta.get("query_lap", -1)) != int(query_lap):
        return False
    if int(meta.get("query_stride", -1)) != int(query_stride):
        return False
    if int(meta.get("top_k", -1)) != int(top_k):
        return False
    return True


def _metrics_and_plots_from_errors_df(
    df,
    *,
    cam: str,
    loc_dir: Path,
    map_mcap: Path,
    localize_times_s: Optional[np.ndarray] = None,
) -> Dict:
    """Compute metrics + err/traj plots from a saved errors table."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(df)
    stamps = df["stamp_ns"].to_numpy(np.int64)
    gt_xy = np.c_[df["gt_x"].to_numpy(float), df["gt_y"].to_numpy(float)]
    gt_yaw = df["gt_yaw"].to_numpy(float)
    ok = df["ok"].astype(bool).to_numpy()
    pred_xy = np.c_[df["pred_x"].to_numpy(float), df["pred_y"].to_numpy(float)]
    pred_yaw = df["pred_yaw"].to_numpy(float)
    err_xy = (
        df["err_xy"].to_numpy(float)
        if "err_xy" in df.columns
        else np.hypot(pred_xy[:, 0] - gt_xy[:, 0], pred_xy[:, 1] - gt_xy[:, 1])
    )
    err_yaw = (
        df["err_yaw_deg"].to_numpy(float)
        if "err_yaw_deg" in df.columns
        else np.full(len(df), np.nan)
    )

    metrics = compute_camloc_metrics(
        stamps_ns=stamps,
        gt_xy=gt_xy,
        gt_yaw=gt_yaw,
        pred_xy=pred_xy,
        pred_yaw=pred_yaw,
        ok=ok,
        localize_times_s=localize_times_s,
        map_mcap=map_mcap,
    )
    metrics["camera"] = cam

    t_rel = (stamps - stamps.min()) * 1e-9
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    axes[0].plot(t_rel, err_xy, color="#c0392b", lw=1.0, marker="o", ms=2)
    axes[0].set_ylabel("err_xy [m]")
    axes[0].set_title(f"camera_{cam} Lap3  success={ok.sum()}/{len(ok)}")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_rel, err_yaw, color="#2980b9", lw=1.0, marker="o", ms=2)
    axes[1].set_ylabel("err_yaw [deg]")
    axes[1].set_xlabel("t [s] relative")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(loc_dir) / "err_vs_time.png", dpi=140)
    plt.close(fig)

    with open(sfm.EXTRACTED / "track_bounds.json") as f:
        tb = json.load(f)
    tl = np.array([[p["x"], p["y"]] for p in tb["track_left"]["points"]], float)
    trb = np.array([[p["x"], p["y"]] for p in tb["track_right"]["points"]], float)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(tl[:, 0], tl[:, 1], color="#7f8c8d", lw=0.7)
    ax.plot(trb[:, 0], trb[:, 1], color="#95a5a6", lw=0.7)
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="#27ae60", lw=1.2, label="GT")
    if ok.any():
        ax.plot(
            pred_xy[ok, 0],
            pred_xy[ok, 1],
            color="#e74c3c",
            lw=1.0,
            marker="o",
            ms=2,
            label="pred",
        )
    ax.set_aspect("equal")
    ax.legend(frameon=False)
    ax.set_title(f"Lap3 traj vs GT (map_{cam} Lap2+4 s-order)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(loc_dir) / "traj_vs_gt.png", dpi=140)
    plt.close(fig)

    return metrics


def run_camera_multi_lap_s_pipeline(
    camera: str,
    *,
    feats: Union[SPNVJointFeatures, HFNetONNX],
    calib: sfm.CamCalib,
    gt,
    laps_map: Dict[int, dict],
    lap_q: dict,
    track_line,
    out_root: Path,
    map_laps: Tuple[int, ...] = (2, 4),
    query_lap: int = 3,
    map_max_frames: Optional[int],
    query_max_frames: Optional[int],
    map_stride: int,
    query_stride: int,
    min_nn_dist: float,
    neighbor_mode: str,
    neighbor_k: int,
    neighbor_offset: int,
    n_max: Optional[float],
    min_baseline_m: float,
    epi_thresh_px: float,
    top_k: int,
    override_map: bool,
    override_query: bool = False,
    polygons_path: Path = DEFAULT_POLYGONS,
    map_mcap: Path = DEFAULT_MAP_MCAP,
) -> Dict:
    """Independent Lap2+Lap4 s-order map + Lap3 query for one camera.

    Caches map in ``map_{cam}_lap2+4/`` and query results in ``loc_lap3_{cam}/``
    (``errors.csv`` + ``query_meta.json``). Same map fingerprint + query lap
    skips re-localization unless ``override_query`` / ``override_map``.
    """
    import pandas as pd

    cam = camera.strip().lower()
    map_dir = Path(out_root) / f"map_{cam}_lap2+4"
    loc_dir = Path(out_root) / f"loc_lap3_{cam}"
    map_dir.mkdir(parents=True, exist_ok=True)
    loc_dir.mkdir(parents=True, exist_ok=True)

    ego_mask = load_ego_mask(cam, polygons_path)
    if ego_mask is not None:
        print(
            f"[{cam}] ego mask from {polygons_path}  "
            f"ignored_px={(ego_mask == 0).mean()*100:.1f}%"
        )
    else:
        print(f"[{cam}] no ego polygons — full frame features")

    skip = sfm.map_ready(map_dir) and not override_map
    lap_ids: Optional[List[int]] = None

    if skip:
        print(f"[{cam}] LOAD map", map_dir / sfm.MAP_BUNDLE_NAME)
        map_frames, points = sfm.load_map_bundle(map_dir)
        csv_path = map_dir / "frames_with_s.csv"
        if csv_path.is_file():
            import pandas as pd

            meta = pd.read_csv(csv_path)
            by_stamp = meta.set_index("stamp_ns")
            s_map = np.array(
                [float(by_stamp.loc[f.stamp_ns, "s"]) for f in map_frames]
            )
            n_map = np.array(
                [float(by_stamp.loc[f.stamp_ns, "n"]) for f in map_frames]
            )
            epsi_map = np.array(
                [float(by_stamp.loc[f.stamp_ns, "epsi"]) for f in map_frames]
            )
            if "lap" in meta.columns:
                lap_ids = [
                    int(by_stamp.loc[f.stamp_ns, "lap"]) for f in map_frames
                ]
        else:
            x = np.array([f.gt_xyzyaw[0] for f in map_frames], float)
            y = np.array([f.gt_xyzyaw[1] for f in map_frames], float)
            yaw = np.array([f.gt_xyzyaw[3] for f in map_frames], float)
            ts = np.array([f.stamp_ns for f in map_frames], np.int64)
            s_map, n_map, epsi_map, _ = track_line.xy_to_sn(x, y, yaw, ts)
    else:
        raw_frames: List[sfm.FrameData] = []
        lap_tag: List[int] = []
        for L in map_laps:
            info = laps_map[L]
            t0 = time.perf_counter()
            frs = sfm.extract_frames_from_mcap(
                sfm.DEFAULT_MCAP,
                int(info["start_ns"]),
                int(info["end_ns"]),
                calib,
                gt,
                camera=cam,
                max_frames=map_max_frames,
                stride=map_stride,
            )
            print(
                f"[{cam}] lap{L} extract {len(frs)} in {time.perf_counter()-t0:.1f}s"
            )
            raw_frames.extend(frs)
            lap_tag.extend([L] * len(frs))

        map_frames, s_map, n_map, epsi_map = project_frames_sn(
            raw_frames, track_line, n_max=n_max
        )
        stamp_to_lap = {fr.stamp_ns: lap for fr, lap in zip(raw_frames, lap_tag)}
        lap_ids = [stamp_to_lap[fr.stamp_ns] for fr in map_frames]
        print(
            f"[{cam}] map frames after s-sort {len(map_frames)} "
            f"dropped {len(raw_frames) - len(map_frames)}"
        )
        write_frames_with_s_csv(
            map_dir / "frames_with_s.csv",
            map_frames,
            s_map,
            n_map,
            epsi_map,
            laps=lap_ids,
        )

        t0 = time.perf_counter()
        run_joint_features(
            map_frames,
            feats,
            min_nn_dist=min_nn_dist,
            ego_mask=ego_mask,
            desc=f"feat-map-{cam}",
        )
        print(f"[{cam}] map features {time.perf_counter()-t0:.1f}s")

        pairs = make_s_neighbor_pairs(
            s_map,
            track_line.lap,
            mode=neighbor_mode,
            max_offset=neighbor_offset,
            k=neighbor_k,
        )
        print(f"[{cam}] neighbor mode={neighbor_mode} pairs={len(pairs)}")

        t0 = time.perf_counter()
        points, _obs = build_map_s_neighbors(
            map_frames,
            feats,
            calib.K,
            pairs=pairs,
            min_baseline_m=min_baseline_m,
            epi_thresh_px=epi_thresh_px,
        )
        print(f"[{cam}] points={len(points)} in {time.perf_counter()-t0:.1f}s")
        sfm.save_map_bundle(map_dir, map_frames, points)
        sfm.save_ply(map_dir / "points3D.ply", points)

    pairs = make_s_neighbor_pairs(
        s_map,
        track_line.lap,
        mode=neighbor_mode,
        max_offset=neighbor_offset,
        k=neighbor_k,
    )

    # Rebuilt map this run ⇒ invalidate query cache
    err_csv = loc_dir / "errors.csv"
    meta_ok = query_cache_valid(
        loc_dir,
        map_dir=map_dir,
        query_lap=query_lap,
        query_stride=query_stride,
        top_k=top_k,
    )
    legacy_ok = (
        err_csv.is_file()
        and not (loc_dir / QUERY_META_NAME).is_file()
        and skip
        and (not override_map)
        and (not override_query)
    )
    skip_query = (not override_query) and (not override_map) and skip and (
        meta_ok or legacy_ok
    )

    if skip_query:
        print(f"[{cam}] LOAD query", err_csv, "(legacy)" if legacy_ok else "")
        df = pd.read_csv(err_csv)
        times_path = loc_dir / LOC_TIMES_NAME
        if times_path.is_file():
            loc_times_a = np.load(times_path)
            loc_total = float(np.sum(loc_times_a))
        else:
            loc_times_a = None
            loc_total = float("nan")
        # ensure meta exists for next runs
        query_meta = {
            "camera": cam,
            "query_lap": int(query_lap),
            "query_stride": int(query_stride),
            "query_max_frames": query_max_frames,
            "top_k": int(top_k),
            "map_laps": list(map_laps),
            "map_fingerprint": map_bundle_fingerprint(map_dir),
            "n_query": int(len(df)),
            "n_success": int(df.ok.astype(bool).sum()),
            "legacy_imported": bool(legacy_ok),
        }
        (loc_dir / QUERY_META_NAME).write_text(json.dumps(query_meta, indent=2))
        metrics = _metrics_and_plots_from_errors_df(
            df,
            cam=cam,
            loc_dir=loc_dir,
            map_mcap=map_mcap,
            localize_times_s=loc_times_a,
        )
        return {
            "camera": cam,
            "map_dir": map_dir,
            "loc_dir": loc_dir,
            "map_frames": map_frames,
            "query_frames": None,
            "points": points,
            "results": None,
            "ego_mask": ego_mask,
            "skipped_map": skip,
            "skipped_query": True,
            "localize_times_s": loc_times_a,
            "localize_total_s": loc_total,
            "metrics": metrics,
            "errors_df": df,
            "s_map": s_map,
            "n_pairs": len(pairs),
            "n_points": int(len(points)),
        }

    t0 = time.perf_counter()
    query_frames = sfm.extract_frames_from_mcap(
        sfm.DEFAULT_MCAP,
        int(lap_q["start_ns"]),
        int(lap_q["end_ns"]),
        calib,
        gt,
        camera=cam,
        max_frames=query_max_frames,
        stride=query_stride,
    )
    print(f"[{cam}] query extract {len(query_frames)} in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    run_joint_features(
        query_frames,
        feats,
        min_nn_dist=min_nn_dist,
        ego_mask=ego_mask,
        desc=f"feat-qry-{cam}",
    )
    print(f"[{cam}] query features {time.perf_counter()-t0:.1f}s")

    results = []
    loc_times = []
    t0 = time.perf_counter()
    for i, q in enumerate(tqdm(query_frames, desc=f"loc-{cam}", unit="qry")):
        t_i = time.perf_counter()
        r = sfm.localize_query(
            q, map_frames, points, feats, calib.K, calib.T_base_cam, top_k=top_k
        )
        loc_times.append(time.perf_counter() - t_i)
        results.append(r)
        if i < 10 or (i + 1) % 50 == 0:
            tqdm.write(
                f"[{cam} {i}] ok={r['ok']} corr={r['n_pnp_corr']} "
                f"inl={r['n_pnp_inl']} err={r['err_xy']}"
            )
    loc_total = time.perf_counter() - t0
    loc_times_a = np.asarray(loc_times, dtype=np.float64)
    print(
        f"[{cam}] localize {loc_total:.1f}s  "
        f"({1000*np.mean(loc_times_a):.1f} ms/frame mean)"
    )

    n_q = len(query_frames)
    pred_xy = np.full((n_q, 2), np.nan)
    pred_yaw = np.full(n_q, np.nan)
    err_xy = np.full(n_q, np.nan)
    err_yaw = np.full(n_q, np.nan)
    for i, r in enumerate(results):
        if r["ok"] and r["pred"] is not None:
            pred_xy[i] = r["pred"][0], r["pred"][1]
            pred_yaw[i] = r["pred"][3]
            err_xy[i] = r["err_xy"]
            err_yaw[i] = r["err_yaw_deg"]

    rows = []
    for i, q in enumerate(query_frames):
        r = results[i]
        rows.append(
            {
                "stamp_ns": q.stamp_ns,
                "ok": bool(r["ok"]),
                "gt_x": q.gt_xyzyaw[0],
                "gt_y": q.gt_xyzyaw[1],
                "gt_yaw": q.gt_xyzyaw[3],
                "pred_x": pred_xy[i, 0],
                "pred_y": pred_xy[i, 1],
                "pred_yaw": pred_yaw[i],
                "err_xy": err_xy[i],
                "err_yaw_deg": err_yaw[i],
                "n_pnp_corr": r["n_pnp_corr"],
                "n_pnp_inl": r["n_pnp_inl"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(loc_dir / "errors.csv", index=False)
    df.loc[df.ok, ["stamp_ns", "pred_x", "pred_y", "pred_yaw"]].to_csv(
        loc_dir / "submission_xy.csv", index=False
    )
    np.save(loc_dir / LOC_TIMES_NAME, loc_times_a)

    query_meta = {
        "camera": cam,
        "query_lap": int(query_lap),
        "query_stride": int(query_stride),
        "query_max_frames": query_max_frames,
        "top_k": int(top_k),
        "map_laps": list(map_laps),
        "map_fingerprint": map_bundle_fingerprint(map_dir),
        "n_query": int(len(df)),
        "n_success": int(df.ok.astype(bool).sum()),
    }
    (loc_dir / QUERY_META_NAME).write_text(json.dumps(query_meta, indent=2))
    print(f"[{cam}] saved query cache", loc_dir / "errors.csv", loc_dir / QUERY_META_NAME)

    metrics = _metrics_and_plots_from_errors_df(
        df,
        cam=cam,
        loc_dir=loc_dir,
        map_mcap=map_mcap,
        localize_times_s=loc_times_a,
    )

    return {
        "camera": cam,
        "map_dir": map_dir,
        "loc_dir": loc_dir,
        "map_frames": map_frames,
        "query_frames": query_frames,
        "points": points,
        "results": results,
        "ego_mask": ego_mask,
        "skipped_map": skip,
        "skipped_query": False,
        "localize_times_s": loc_times_a,
        "localize_total_s": float(loc_total),
        "metrics": metrics,
        "errors_df": df,
        "s_map": s_map,
        "n_pairs": len(pairs),
        "n_points": int(len(points)),
    }
