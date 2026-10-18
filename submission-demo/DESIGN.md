# submission-demo — design document

> **User guide (English):** [`README.md`](README.md) — how to configure mcap paths and run [`run_submission.ipynb`](run_submission.ipynb).  
> **Status:** Architecture reference (implementation matches this doc).  
> **Goal:** Self-contained submission localization under `submission-demo/`: raw **`.mcap`** in, **eval** (training + GT) or **predict** (test), three options (motion-only / two-camera / EKF), online SfM localize with disk cache.
>
> **v2 revision (review):** Calibration / GT / motion state **no longer assume an external `extracted/` tree**—everything is extracted from the input `.mcap` via `tools/` scripts (`extract_assets.py` → camera_info/tf_static/track_bounds/state_estimation.csv; `extract_gt.py` → gt.csv, training only).
>
> **v3 revision:** The repo currently has **no HF-Net weights**; this version **defaults to the `sp_nv_nn` (SuperPoint + NetVLAD + mutual-NN) backend**, matching existing `map_*_lap2+4` descriptors, **reusing ready-made maps without rebuilding**. HF-Net remains an optional future upgrade (interface reserved: `sfm_features.make_features("hfnet_onnx")`; upgrade requires HF-Net remapped bundles + σ(s) recalibration).

---

## 1. Requirements alignment (from conventions)

| # | Requirement | Design placement |
|---|------|---------|
| 1 | Setup can specify training data path and test data path | `SETUP` block: `TRAIN_MCAP` / `TEST_MCAP` (raw `.mcap`). **Calibration/GT/motion extracted from mcap via `tools/`, cached under `outputs/<...>/assets/`**—no separate `extracted/` directory |
| 2 | Setup chooses **evaluation (training-based)** or **prediction (test-based)** | `MODE = "eval" \| "predict"`; `eval` → metrics + submission, `predict` → **submission only, no metrics** |
| 3 | Setup can specify laps to run; default all laps | `LAPS = None` (all) or `LAPS = [3]` / `[5,7]`; see §6 lap windows and test without GT |
| 4 | Three options map to one cell each | Option 1 **motion prediction only**; Option 2 **two-camera localization (fr+rl offline se2 fusion)**; Option 3 **EKF fusion (SE predict + fr/rl extrapolation → se2 → EKF update with R(s))** |
| 5 | Three empty dirs `sfm-model` / `track-model` / `error-model` hold models/assets for self-containment | See §4 layout and §5 copy/rewrite checklist |

**Self-containment:** Do not `import` modules from `3d-map-test/` or `query-demo/`; **rewrite** needed logic as `.py` under `submission-demo/` (reference existing code, do not copy `.ipynb` wholesale). Copy data assets (map bundles, racing line, σ(s) model, calibration) into `sfm-model/` / `track-model/` / `error-model/`.

---

## 2. Background: existing pipeline

Reference `query-demo/lap3_ekf_2cam_fr_rl_query_demo.ipynb` + `query_demo_ekf.py`. End-to-end:

```
Raw mcap ──(per-camera decode + undistort)──> query frames
   │
   └─ NetVLAD top-K on map_{cam}_lap2+4 → mutual NN → PnP  ← online SfM localize
         └─ per camera errors.csv: stamp_ns, ok, pred_x/y/yaw, n_pnp_inl, (gt_* eval only)
                │
   ┌────────────┴─────────────────────────────────────────────┐
   │                        │                                   │
Option1 motion-only     Option2 two-camera se2 fusion      Option3 EKF(2cam)
 SE dead reckoning       nearest-stamp ω-weighted SE(2)    SE predict + fr/rl extrap to t_last
 sample at t_last        (no EKF)                           → se2 → EKF update, R(s)=σ²(s)
   │                        │                                   │
   └─────────── each outputs submission.csv (timestamp_ns,x,y,yaw) ─┘
                            │
                    eval: evaluate.py / compute_camloc_metrics
                    predict: write submission.csv only
```

Differences vs the notebook (new in this module):
- **Extract from mcap:** notebook assumed pre-built `extracted/`; this module runs `tools/extract_assets.py` / `tools/extract_gt.py` at runtime into `outputs/<...>/assets/`.
- **Online localize (`sp_nv_nn`):** notebook read precomputed `loc_lap3_{fr,rl}/errors.csv`; this module runs `sp_nv_nn` on raw mcap against existing `map_{fr,rl}_lap2+4` bundles.
- **Cache reuse:** same task fingerprint (mcap + map + lap + stride + top_k) → second run reads cached `errors.csv` (`query_cache_valid` + `query_meta.json`).
- **Test without GT:** predict mode skips `gt.csv`, no error metrics, no GT lap cropping.

---

## 3. Run mode semantics

### 3.1 `MODE = "eval"` (training data)
- Input: `TRAIN_MCAP`. First run **extracts** via `extract_assets.py` → camera_info / tf_static / track_bounds / state_estimation.csv; `extract_gt.py` → gt.csv (from `/tf`). Cache: `outputs/eval_<stem>/assets/`.
- Lap windows: from `gt.csv` (start/finish plane crossing, `lapdata` logic) or `lapdata/laps.json` for same mcap.
- Frames: `gt.csv` fills `gt_xyzyaw` for `localize_query` err_xy and official eval.
- Outputs: per option `submission.csv` + `metrics_official.csv` (`compute_camloc_metrics`) + plots + `compare/` summary. Optional `tools/evaluate.py` (x/y and s/n).

### 3.2 `MODE = "predict"` (test data)
- Input: `TEST_MCAP`. `extract_assets.py` only—**no gt.csv** (test has no `/tf` GT).
- Frames: `gt_xyzyaw` placeholder `(nan,nan,nan,nan)`; does not affect localization, only disabled err fields.
- Outputs: per option `submission.csv` only—**no** metrics, GT crop, or evaluate.

> Key 1: `localize_query` computes `err_xy = hypot(pred-gt)`; with nan GT, err is nan; `ok` still from PnP inliers (no throw).
>
> Key 2: `/constructor0/state_estimation` (vx/vy/ax/ay/wz + pose) exists on **training and test** (on-vehicle SE, not `/tf` GT). Options 1/3 motion input works in predict; init from SE first pose.

---

## 4. Target directory layout

```text
submission-demo/
  DESIGN.md                     # this file
  README.md                     # after codegen: how to run
  run_submission.ipynb          # SETUP + 3 option cells + summary (eval)
  config/
    process_noise.yaml          # EKF process noise (from error-model se_step_residual_noise.json)

  tools/                        # [existing] mcap extract + eval (reuse, not rewritten)
    extract_assets.py           # camera_info/tf_static/track_bounds/state_estimation.csv
    extract_gt.py               # gt.csv from /tf, training only
    convert.py, evaluate.py, track_frame.py

  assets_io.py                  # wrap tools/extract_*; cache/load assets
  sfm_core.py                   # self-contained SfM core (localize subset from sfm_experiment.py)
  sfm_features.py               # features (from sfm_hfnet.py: SPNVJointFeatures + NN; HFNetONNX stub)
  sfm_localize.py               # online localize + cache (query half of multi-lap-s pipeline)
  mcap_io.py                    # mcap frames (from camloc_play.py)
  obs_sigma.py                  # σ(s) (from obs_error_sigma_s.py)
  fusion.py                     # se2 fusion + confidence (from sfm_hfnet.py fuse_*)
  motion_ekf.py                 # SE DR + SE2EKF + extrap (from query_demo_ekf.py)
  metrics.py                    # compute_camloc_metrics (eval)
  pipeline.py                   # orchestration: assets → localize(cache) → three options → disk

  sfm-model/                    # [assets] sp_nv_nn prebuilt map bundles (load only)
    map_fr_lap2+4/              # map_bundle.npz, netvlad_db.npz, frames_with_s.csv, points3D.ply
    map_rl_lap2+4/
    models/
      README_HFNET.md           # future HF-Net upgrade; no hfnet.onnx in this version
    polygons.json               # fr/rl ego mask (from fr-auto-mask/polygons.json)

  track-model/                  # [assets] racing line
    map.mcap                    # from 23-00-38_cut_120m_update/map.mcap (track_frame)

  error-model/                  # [assets] error / noise models
    obs_error_sigma_vs_s.json   # σ(s) (from loc_lap3_fused/; recalibrate after HF-Net)
    se_step_residual_noise.json # process noise MADσ (from state-error-estimation/)

  outputs/
    <mode>_<mcap_stem>/
      assets/                   # extracted camera_info/.../gt.csv (eval)
      loc/loc_<lap>_fr/errors.csv (+ query_meta.json, localize_times_s.npy)
      loc/loc_<lap>_rl/errors.csv ...
      motion_only/submission.csv [+ metrics_official.csv eval]
      two_cam/submission.csv      [+ metrics_official.csv eval]
      ekf/submission.csv          [+ metrics_official.csv eval, sim/ekf_sim.csv]
      compare/metrics_compare.csv, *.png   (eval only)
      gt_lap<lap>.csv             (eval only, cropped GT for evaluate)
```

> **Backend = `sp_nv_nn` (this version):** default `SPNVJointFeatures` (SuperPoint local + NetVLAD global), mutual-NN at query (no LightGlue). Matches `map_*_lap2+4` descriptors—**reuse maps, no rebuild**.
>
> **HF-Net upgrade stub:** `make_features("hfnet_onnx")` / `HFNetONNX` in `sfm_features.py`; switching needs remapped bundles + σ(s) recalibration (§14).

---

## 5. Copy or rewrite checklist (self-contained)

| New file | Source (reference) | Functions / classes | External deps |
|--------|-----------|----------------|----------|
| `assets_io.py` | `tools/extract_assets.py` + `extract_gt.py` (call, don’t rewrite) | `extract_assets`, `extract_gt`, `load_assets`; existence cache | `mcap`, `mcap_ros2` |
| `mcap_io.py` | `camloc_play.py` | `camera_topic`, `stamp_ns_from_msg`, `decode_jpeg`, `ImageFrame`, `iter_camera_frames` | `mcap`, `mcap_ros2`, `cv2`, `numpy` |
| `sfm_core.py` | `sfm_experiment.py` | `pick_device`, transforms, `CamCalib`, `undistort`, `localize_query`, `load_map_bundle`, … | `torch`, `lightglue`, `scipy`, `cv2`, `pandas`, `numpy` |
| `sfm_features.py` | `sfm_hfnet.py` | **Main** `SPNVJointFeatures`, `NNMatcher`, `mutual_nn_match`, `make_features`, ego mask helpers; **stub** `HFNetONNX` | `cv2`, `numpy`, `torch`, `lightglue`; + `onnxruntime` for HF-Net |
| `sfm_localize.py` | `sfm_hfnet.py` query + cache | `localize_camera`, fingerprints, `query_cache_valid` | `sfm_core`, `sfm_features` |
| `fusion.py` | `sfm_hfnet.py` / `query_demo_ekf.py` | `pose_confidence`, `align_by_nearest_stamp`, `fuse_se2_weighted`, `fuse_loc_tables_se2` (Option 2) | `numpy`, `pandas` |
| `motion_ekf.py` | `query_demo_ekf.py` | `dead_reckon_se`, `SE2EKF`, `run_mode_a_se_predict`, `run_mode_b_ekf`, `write_submission`, … | `numpy`, `pandas`, `yaml` |
| `obs_sigma.py` | `obs_error_sigma_s.py` | `load_params`, `sigma_of_s` → `error-model/obs_error_sigma_vs_s.json` | `numpy` |
| `metrics.py` | `sfm_hfnet.py` | `compute_camloc_metrics` (Option B via `tools/track_frame`), eval only | `numpy`, `pandas` |
| `pipeline.py` | new orchestration | `SubmissionConfig`, `ensure_assets`, `run_motion_only`, `run_two_cam`, `run_ekf`, `run_all` | all above |

> `track_frame.py` is not rewritten—use `tools/track_frame.py` (`sys.path` import `load_lines`).

### 5.1 Maps: reuse existing `sp_nv_nn` bundles (no mapping in v1)

`3d-map-test/outputs_hfnet_multi_lap_s/map_{fr,rl}_lap2+4/` matches this backend—**copy into `sfm-model/`**. `sfm_localize.localize_camera` loads for query.

> No mapping code (`project_frames_sn` / `build_map_s_neighbors` / `triangulate_pair`, etc.). Add `sfm_mapping.py` later if HF-Net or new laps need rebuild (§14).

---

## 6. Lap window sources (and test without GT)

`lapdata/laps.json` windows use **GT start/finish plane crossing**.

- **eval:** with `gt.csv` → use `laps.json` or recompute. `LAPS=None` all laps; `LAPS=[3]` lap 3 only. Each lap `[start_ns,end_ns]` for frames + `gt_lap{N}.csv` for evaluate.
- **predict (test):** **no GT → no plane-crossing lap split**. Options:
  1. `LAPS=None` (default): **whole mcap**, one `submission.csv`.
  2. Manual windows: `TEST_WINDOWS=[(start_ns,end_ns), ...]`.
  3. External `laps.json`: `TEST_LAPS_JSON` path.
- Default predict = strategy 1; README documents 2/3.

---

## 7. SETUP cell specification

```python
# ---- paths (raw mcap; calib/GT/SE extracted from mcap) ----
TRAIN_MCAP = ".../c0_..._0.mcap"      # training mcap (with /tf GT)
TEST_MCAP  = ".../test_xxx.mcap"      # test mcap (no /tf GT)

# ---- run target ----
MODE   = "eval"           # "eval" (training+GT, metrics) | "predict" (test, submission only)
LAPS   = None             # None=all; or [3] / [5,7]
WINDOWS    = None         # optional [(start_ns,end_ns),...] when predict has no GT laps
LAPS_JSON  = None         # optional external lap windows json

# ---- cameras / backend ----
CAMS       = ("fr", "rl")           # two-camera main path
REF_CAMERA = "fr"                   # sync anchor
BACKEND    = "sp_nv_nn"             # mainline; "hfnet_onnx" needs weights + remap (future)

# ---- localize params ----
QUERY_STRIDE = 1
TOP_K        = 5
MAP_LAPS     = (2, 4)               # fixed in bundle
OVERRIDE_QUERY  = False             # True=ignore query cache

# ---- resources (self-contained defaults under submission-demo) ----
SFM_MODEL_DIR = "sfm-model"
TRACK_MCAP    = "track-model/map.mcap"
SIGMA_JSON    = "error-model/obs_error_sigma_vs_s.json"
NOISE_JSON    = "error-model/se_step_residual_noise.json"
TOOLS_DIR     = "tools"
OUT_ROOT      = "outputs"
```

Derived: `DATA_MCAP = TRAIN_MCAP if MODE=="eval" else TEST_MCAP`; `HAS_GT = (MODE=="eval")`; `ASSETS_DIR = outputs/<mode>_<stem>/assets`.

---

## 8. Three option cells

All options ensure **visual observations ready** (per-camera `errors.csv`, online or cache). Option 1 does not need vision but shares the two-camera timeline (sample at `t_last` via sync buckets).

### Option 1 — motion prediction only (`motion_only/`)
- `dead_reckon_se`: init from first pose (eval: GT; predict: SE first frame).
- Sample at sync `t_last` → `submission.csv`.
- eval: metrics + plots; predict: write only.
- **Requires `state_estimation.csv`** on test—document prerequisite.

### Option 2 — two-camera localization (`two_cam/`)
- Ensure `loc_<lap>_fr/errors.csv` and `loc_<lap>_rl/errors.csv`.
- `fuse_loc_tables_se2(tables, ref_camera="fr")`: nearest-stamp ω-weighted SE(2) (gate Δxy≤5m, Δyaw≤15°), no EKF.
- `ok` rows → `submission.csv`. eval metrics use fused `gt_*`.

### Option 3 — EKF fusion (`ekf/`)
- `run_mode_b_ekf`: SE2EKF; per bucket `predict_to(t_last)` → extrapolate fr/rl PnP → `fuse_se2_weighted` → `observation_R(s)=diag(σ²(s),σ²(s),σ_yaw²)` → `update`.
- Process noise from `config/process_noise.yaml` (MADσ from `se_step_residual_noise.json`).
- `submission.csv` + `sim/ekf_sim.csv`. eval metrics.

### Summary cell (eval only)
- `compare/metrics_compare.csv`; trajectory/error plots.
- predict: print paths and row counts for three submissions only.

---

## 9. Online localize + cache (`sfm_localize.localize_camera`)

```text
localize_camera(cam, *, data_mcap, extracted, lap_window, out_loc_dir, map_dir,
                feats, calib, gt_or_none, track_line, query_stride, top_k,
                map_laps, override_query, polygons_path, has_gt):
  1. map_frames, points = load_map_bundle(map_dir)
  2. fp = map_bundle_fingerprint(map_dir)
     if (not override_query) and query_cache_valid(out_loc_dir, fp, lap, stride, top_k):
         return read_csv(out_loc_dir/errors.csv)
  3. frames = extract_query_frames(...)
  4. run_joint_features(frames, feats, ego_mask=...)
  5. for q in frames: localize_query(...); append rows
  6. df.to_csv(errors.csv); query_meta.json; localize_times_s.npy
  7. return df
```

Cache key in `query_meta.json`: map fingerprint + **data_mcap fingerprint (new)** + `query_lap` + `query_stride` + `top_k`—train/test mcap must not cross-hit cache.

errors.csv schema:
```
stamp_ns, ok, gt_x, gt_y, gt_yaw, pred_x, pred_y, pred_yaw, err_xy, err_yaw_deg, n_pnp_corr, n_pnp_inl
```
predict: `gt_*` and `err_*` are `nan`.

---

## 10. Resource copy list (at codegen time)

| Target | Source | Notes |
|------|------|------|
| `sfm-model/map_fr_lap2+4/`, `map_rl_lap2+4/` | `3d-map-test/outputs_hfnet_multi_lap_s/map_{fr,rl}_lap2+4/` (all 4 files) | sp_nv_nn maps |
| `sfm-model/polygons.json` | `fr-auto-mask/polygons.json` | ego mask |
| `sfm-model/models/README_HFNET.md` | `3d-map-test/models/README_HFNET.md` | HF-Net upgrade notes |
| `track-model/map.mcap` | `c0_.../23-00-38_cut_120m_update/map.mcap` | racing line |
| `error-model/obs_error_sigma_vs_s.json` | loc_lap3_fused copy | sp_nv_nn calibration |
| `error-model/se_step_residual_noise.json` | `state-error-estimation/se_step_residual_noise.json` | process noise |

> Large files (`map.mcap`, `map_bundle.npz` ~400 MB/camera). Consider `.gitignore`, symlinks, or runtime paths—confirm copy vs symlink before codegen.

---

## 11. How to run (README draft)

```bash
# deps: torch, lightglue, numpy, pandas, opencv-python, scipy, pyyaml,
#       mcap, mcap-ros2, tqdm, (optional onnxruntime)
# 1) Edit SETUP in run_submission.ipynb (MODE / paths / LAPS)
# 2) Run SETUP → Option1 → Option2 → Option3 → (eval) summary
# 3) outputs/<mode>_<mcap>/{motion_only,two_cam,ekf}/submission.csv
```
First run: online SfM localize (slow). Same task again: read `outputs/.../loc/.../errors.csv` cache (fast).

---

## 12. Open questions

1. **Deliverable shape:** notebook main entry—also a thin CLI `run_submission.py`?
2. **Large files:** copy vs symlink for `map_bundle.npz` / `map.mcap`?
3. **Lap split (predict):** default whole mcap (§6 strategy 1) OK, or will you supply `WINDOWS` / `LAPS_JSON`?

---

## 13. Implementation order (after sign-off)

1. Place assets: maps, map.mcap, σ(s), noise, polygons.
2. Low level: `assets_io.py`, `mcap_io.py`, `obs_sigma.py`.
3. SfM: `sfm_core.py` → `sfm_features.py` → `sfm_localize.py` (cache, existing maps).
4. Localization: `fusion.py`, `motion_ekf.py`, `metrics.py`.
5. Orchestration: `pipeline.py` + `config/process_noise.yaml`.
6. Entry: `run_submission.ipynb` (+ optional CLI).
7. Self-test: eval training lap3 vs notebook scale; predict submission on test mcap.

---

## 14. Future upgrade: HF-Net (not in this version)

No `hfnet.onnx` in repo; maps are `sp_nv_nn`. HF-Net upgrade requires:

1. **Weights:** export `hfnet.onnx` per `README_HFNET.md` into `sfm-model/models/`.
2. **Maps:** HF-Net descriptors incompatible—**rebuild** `map_{fr,rl}_lap2+4` (`sfm_mapping.py`: frames → HF-Net → s-order triangulation → `save_map_bundle`).
3. **σ(s) recalibration:** `error-model/obs_error_sigma_vs_s.json` from HF-Net fused errors.

`sfm_features.make_features("hfnet_onnx")` / `HFNetONNX` stubbed; switch after the three assets above.
