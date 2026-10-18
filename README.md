# PixelRacers

Constructor Labs track **camera localization**: mcap playback, HF-Net / sp_nv_nn SfM mapping & localization, state-estimation process noise, Lap3 EKF query-demo, and a **self-contained submission pipeline** in [`submission-demo/`](submission-demo/).

---

## Step 1 — Set up the environment

Complete these steps once per machine before running notebooks or scripts.

### 1.1 Clone the repository (with submodules)

**Hierarchical Localization (hloc)** in `third_party/Hierarchical-Localization` is required (NetVLAD, localize stack):

```bash
git clone --recurse-submodules git@github.com:baihao/PixelRacers.git
cd PixelRacers
```

If you already cloned without submodules:

```bash
cd PixelRacers
git submodule update --init --recursive
```

Upstream: [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization).

### 1.2 Git LFS — submission maps and track model

Prebuilt SfM bundles and `track-model/map.mcap` are stored with **Git LFS**:

```bash
brew install git-lfs    # or see https://git-lfs.com
git lfs install         # once per machine
git lfs pull            # from repo root (right after clone)
```

Check that files are real binaries (not ~130-byte pointers), e.g. `submission-demo/sfm-model/map_fr_lap2+4/map_bundle.npz`.

### 1.3 Python virtual environment

From the **repo root**:

```bash
bash scripts/setup_pixel_racers_venv.sh
source scripts/env.sh
```

This creates **`pixel-racers/`** (gitignored), runs `git submodule update --init --recursive`, installs [`requirements.txt`](requirements.txt), and registers Jupyter kernel **`pixel-racers`**.

| Flag | Meaning |
|------|---------|
| `--recreate` | Delete existing `pixel-racers/` venv first |
| `--no-kernel` | Skip `ipykernel` registration |
| `--skip-submodule` | Skip submodule init (not recommended on fresh clone) |

Manual equivalent: `python3.12 -m venv pixel-racers` → activate → submodule init → `pip install -r requirements.txt` → `ipykernel install`.

On Apple Silicon, SfM notebooks expect **MPS**. On macOS OpenMP conflicts: `export KMP_DUPLICATE_LIB_OK=TRUE` (set in `scripts/env.sh`).

### 1.4 Third-party software and models

Dependencies include **hloc**, **LightGlue / SuperPoint**, **PyTorch**, **NetVLAD (Pitts30K weights, downloaded on first localize run)**, **OpenCV**, **mcap**, and others. See **[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)** before redistribution.

Optional HF-Net ONNX (not required for submission demo): [`3d-map-test/models/README_HFNET.md`](3d-map-test/models/README_HFNET.md). Submission default: **`sp_nv_nn`**.

### 1.5 Challenge `.mcap` bags (not in git)

Download training and/or test bags separately. Example paths on your disk:

```text
/path/to/c0_..._camloc/c0_..._0.mcap              # eval
/path/to/rosbag_..._cameras_lap/..._0.mcap        # predict
```

No pre-built `extracted/` folder is required for **`submission-demo`** — assets are extracted from the mcap on first run.

---

## Step 2 — Test with `submission-demo`

The recommended verification path is **[`submission-demo/run_submission.ipynb`](submission-demo/run_submission.ipynb)** (kernel **`pixel-racers`**).

```bash
cd PixelRacers/submission-demo
jupyter notebook run_submission.ipynb
```

**Full walkthrough** (config table, cell order, outputs): **[`submission-demo/README.md`](submission-demo/README.md)**.

### 2.1 Configure mcap paths

Edit the **User configuration** cell — use **absolute paths**:

| Setting | Eval (training + GT) | Predict (test, no GT) |
|---------|----------------------|------------------------|
| `MODE` | `"eval"` | `"predict"` |
| Mcap | `TRAIN_MCAP = "/path/to/training_0.mcap"` | `TEST_MCAP = "/path/to/test_0.mcap"` |
| Laps | `LAPS = [5, 6, 7]` or `None` | N/A (full bag unless manual windows) |
| Initial pose | ignored (GT used) | `INIT_POSE_XYYAW = (x, y, yaw)` |

### 2.2 Run notebook cells

User config → SETUP → Option 1 → Option 2 → Option 3 → Summary → Deliverables.

Example artifacts: [`submission-files/`](submission-files/).

### Reference performance (two-camera localize, Option 2 / 3)

Measured on **MacBook Air, Apple M3, 2024** with **`sp_nv_nn`**, `QUERY_STRIDE = 1`:

- **~100 ms per image** per camera (fr / rl): preprocess → SuperPoint + NetVLAD → retrieval → PnP (`loc_ms_mean_fr` / `loc_ms_mean_rl` in eval tables).
- **Overall ~2 FPS** end-to-end: dual-camera localize plus mcap I/O, **SE(2) fusion** (Option 2) or **EKF** (Option 3). Option 1 (motion only) is much faster. Cached `loc/.../errors.csv` on re-runs.

---

## Components

| Path | Role |
|------|------|
| [`submission-demo/`](submission-demo/) | **Submission pipeline** — start here for testing |
| [`submission-files/`](submission-files/) | Example submission + evaluate report |
| `3d-map-test/` | Research: Lap2+4 map + Lap3 loc + σ(s) |
| `query-demo/` | Research: SE vs EKF query demos |
| `odometry/`, `state-error-estimation/` | Baselines and process noise |
| `camloc_play.py` | mcap camera preview |
| `third_party/Hierarchical-Localization` | hloc submodule |

---

## Research workflow (optional notebooks)

After Step 1–2, explore:

| Area | Entry |
|------|--------|
| Play mcap | `camloc_play.py`, `test/play_mcap.ipynb` |
| Odometry | `odometry/lap2_velocity_odometry.ipynb` |
| Process noise | `state-error-estimation/se_step_residual_noise.ipynb` |
| SfM + Lap3 loc | `3d-map-test/sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb` |
| EKF query-demo | `query-demo/lap3_ekf_query_demo.ipynb`, `lap5_ekf_query_demo.ipynb` |

Evaluate with lap-cropped GT:

```bash
python submission-demo/tools/evaluate.py \
  query-demo/outputs/lap3/gt_lap3.csv \
  query-demo/outputs/lap3/ekf_cam/submission.csv
```

---

## FAQ

| Issue | Fix |
|-------|-----|
| Missing `lightglue` / hloc | `git submodule update --init --recursive`; rerun `setup_pixel_racers_venv.sh` |
| OpenMP duplicate library | `export KMP_DUPLICATE_LIB_OK=TRUE` |
| Maps are tiny text files | At repo root: `git lfs pull` (after clone or after `git pull` when maps were updated) |
| evaluate coverage &lt; 90% | Use lap-cropped GT, not full-trip `gt.csv` |
| query-demo missing loc / σ(s) | Run 3d-map-test notebook through fused σ(s) export |
