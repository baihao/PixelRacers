# submission-demo

Self-contained **two-camera (fr + rl)** localization submission pipeline. Input is a raw **`.mcap`** ROS 2 bag. Supports **eval** (training bag + ground truth → metrics + submissions) and **predict** (test bag → submissions only). Three options: motion-only dead reckoning, two-camera visual fusion, and EKF fusion.

Design notes: [`DESIGN.md`](DESIGN.md).

---

## Quick test (recommended): `run_submission.ipynb`

This is the **primary entry point** for reviewers and developers.

### 1. Prerequisites

1. Clone **PixelRacers** and run **`bash scripts/setup_pixel_racers_venv.sh`** (see [root `README.md`](../README.md) § Environment), then `source scripts/env.sh`.
2. Jupyter kernel: **`pixel-racers`**.
3. **Git LFS** maps (if not already on disk):

   ```bash
   cd PixelRacers
   git lfs install
   git lfs pull
   ```

   You need real files under `sfm-model/map_*_lap2+4/` (`*.npz`, `*.ply`) and `track-model/map.mcap` — not tiny pointer text files.

4. A **training** or **test** `.mcap` on your machine (not in git). Example layout:

   ```text
   /path/to/data/
     c0_2026-10-13_22-53-42_camloc/
       c0_2026-10-13_22-53-42_camloc_0.mcap    # eval / training
     rosbag_2026-10-13_16-08-46_cameras_lap/
       rosbag_2026-10-13_16-08-46_cameras_lap_0.mcap   # predict / test
   ```

   No pre-extracted `extracted/` folder is required: the notebook **extracts** camera calib, tf, track bounds, state estimation, and (eval only) GT from the mcap into `outputs/.../assets/`.

### 2. Open the notebook

```bash
cd PixelRacers/submission-demo
jupyter notebook run_submission.ipynb
# or open run_submission.ipynb in Cursor / VS Code with kernel pixel-racers
```

### 3. Edit the **User configuration** cell (first code cell after the title)

Set **absolute paths** to your mcap files and the run mode:

| Variable | Purpose |
|----------|---------|
| `TRAIN_MCAP` | Path to **training** `.mcap` (has `/tf` GT). Used when `MODE="eval"`. |
| `TEST_MCAP` | Path to **test** `.mcap` (no GT). Used when `MODE="predict"`. |
| `MODE` | `"eval"` → use `TRAIN_MCAP`, compute metrics. `"predict"` → use `TEST_MCAP`, submissions only. |
| `LAPS` | **eval only**: which laps to run, e.g. `[5, 6, 7]`, or `None` for all complete laps. Ignored for predict (whole bag unless you set windows). |
| `LAPS_JSON` | Optional path to lap time windows, e.g. `../lapdata/laps.json`. `None` = detect laps from GT in eval. |
| `INIT_POSE_XYYAW` | **predict only**: initial `(x, y, yaw)` in map frame. **Ignored in eval** (GT first pose is used). |
| `INIT_POSE_T0_NS` | Optional timestamp for init pose; `None` = default from SE / GT. |

**Example — eval on training bag, laps 5–7:**

```python
TRAIN_MCAP = "/path/to/c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap"
TEST_MCAP  = "/path/to/rosbag_2026-10-13_16-08-46_cameras_lap/rosbag_2026-10-13_16-08-46_cameras_lap_0.mcap"
MODE = "eval"
LAPS = [5, 6, 7]
LAPS_JSON = "/path/to/PixelRacers/lapdata/laps.json"
INIT_POSE_XYYAW = (0.0, 0.0, 0.0)   # ignored in eval
INIT_POSE_T0_NS = None
```

**Example — predict on test bag:**

```python
MODE = "predict"
# TRAIN_MCAP still set but unused; pipeline reads TEST_MCAP:
TEST_MCAP = "/path/to/rosbag_2026-10-13_16-08-46_cameras_lap/rosbag_2026-10-13_16-08-46_cameras_lap_0.mcap"
INIT_POSE_XYYAW = (-120.0, 695.0, -1.842)   # required for predict unless using run_init_pose.ipynb
```

For help finding a good initial pose on test data, use [`run_init_pose.ipynb`](run_init_pose.ipynb).

### 4. Run cells in order

| Step | Cell | What it does |
|------|------|----------------|
| 1 | **User configuration** | Paths and mode (above). |
| 2 | **SETUP** | Builds `SubmissionConfig`, extracts assets from mcap, loads maps, defines windows. |
| 3 | **Option 1** | Motion-only dead reckoning. |
| 4 | **Option 2** | fr+rl localize + SE(2) fusion (slow first time; cached after). |
| 5 | **Option 3** | EKF fusion. |
| 6 | **Summary** | Overlay trajectories + metric bars (`window=all`). |
| 7 | **Deliverables** | Writes `compare/evaluate_table.csv` (eval). |

After each option cell you should see a **combined-lap figure** and a **vertical metrics table** from `tools/evaluate.py` (eval mode).

**First run** on Option 2/3 can take a long time (online feature extraction + PnP). Re-runs reuse `outputs/.../loc/.../errors.csv` when fingerprints match.

**Smoke test (faster):** in SETUP, set `QUERY_STRIDE = 20` (or use CLI `--query-stride 20`).

### 5. Where outputs go

```text
outputs/<mode>_<mcap_stem>/
  assets/                    # extracted from mcap (cached)
  loc/<lap>/loc_{fr,rl}/     # localization cache
  <lap>/{motion_only,two_cam,ekf}/submission.csv
  compare/                   # merged eval tables and summary plots
```

Official scoring: `tools/evaluate.py` (same rules as the vertical table in the notebook).

---

## Backend

Default **`sp_nv_nn`**: SuperPoint local features + NetVLAD global + mutual-NN matching, aligned with bundled maps `sfm-model/map_{fr,rl}_lap2+4/`. Query preprocessing: **CLAHE (Lab-L)** + highlight keypoint filter (`PixelRacers/image_preprocess.py`). HF-Net is a future upgrade ([`DESIGN.md`](DESIGN.md) §14, [`sfm-model/models/README_HFNET.md`](sfm-model/models/README_HFNET.md)).

## Dependencies

```
torch, lightglue, numpy, pandas, opencv-python, scipy, pyyaml,
mcap, mcap-ros2-support, tqdm
```

Kernel: **`pixel-racers`**. NetVLAD weights come from `third_party/Hierarchical-Localization` (Pitts30K download on first use).

### Git LFS

Large assets: `sfm-model/map_*/{map_bundle,netvlad_db}.npz`, `points3D.ply`, `track-model/map.mcap`. After clone:

```bash
git lfs install
git lfs pull    # from PixelRacers repo root
```

Small files (`build_meta.json`, `error-model/`, `config/`, etc.) are normal git objects. Maintainer push: `GIT_LFS_SKIP_PUSH=1 git push` then `git lfs push` (see root `README.md`).

On macOS, if you see `OMP: Error #15`: `export KMP_DUPLICATE_LIB_OK=TRUE` (set automatically in notebook entry).

---

## CLI (optional)

```bash
cd submission-demo
export KMP_DUPLICATE_LIB_OK=TRUE

python run_submission.py --mode eval --train-mcap /path/c0_..._0.mcap
python run_submission.py --mode eval --train-mcap ... --laps 3 --options ekf
python run_submission.py --mode eval --train-mcap ... --laps 3 --query-stride 20
python run_submission.py --mode predict --test-mcap /path/test.mcap
```

---

## Three options

| option | Output dir | Description |
|--------|------------|-------------|
| `motion_only` | `.../motion_only/` | SE dead reckoning sampled at two-camera `t_last` |
| `two_cam` | `.../two_cam/` | fr+rl localize → ω-weighted SE(2) fusion (no EKF) |
| `ekf` | `.../ekf/` | SE predict + fr/rl extrapolation → se2 → EKF, R(s)=σ(s)² |

Each writes `submission.csv` (`timestamp_ns,x,y,yaw`). Eval also writes per-lap metrics and merged `compare/evaluate_table.csv`.

## Bundled resources

| Directory | Contents |
|-----------|----------|
| `sfm-model/map_{fr,rl}_lap2+4/` | Prebuilt sp_nv_nn map bundles |
| `sfm-model/polygons.json` | fr/rl ego masks |
| `track-model/map.mcap` | Racing line for (s,n) / plots |
| `error-model/` | σ(s) and process noise MADσ |
| `config/process_noise.yaml` | Generated from noise JSON on first EKF run |
| `tools/` | `extract_assets.py`, `extract_gt.py`, `evaluate.py`, … |

## Predict (test) notes

- No GT → no metrics, no lap splitting by default: one submission over the full mcap. Use manual `WINDOWS` in SETUP or `LAPS_JSON` if you have external windows.
- `state_estimation` (vx, vy, ax, ay, wz) exists on test bags (vehicle SE, not GT), so Options 1 and 3 work; set **`INIT_POSE_XYYAW`** or run `run_init_pose.ipynb`.
