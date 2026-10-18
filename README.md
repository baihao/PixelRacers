# PixelRacers

Constructor Labs track **camera localization**: mcap playback, HF-Net / sp_nv_nn SfM mapping & localization, state-estimation process noise, Lap3 EKF query-demo, and a **self-contained submission pipeline** in [`submission-demo/`](submission-demo/).

---

## Quick start — test the submission pipeline

The fastest way to verify the repo is to run **`submission-demo/run_submission.ipynb`** on a training or test `.mcap`.

1. **Environment** (once):

   ```bash
   cd PixelRacers
   python3.12 -m venv pixel-racers
   source pixel-racers/bin/activate
   python -m pip install -U pip
   git submodule update --init --recursive
   python -m pip install -r requirements.txt
   python -m ipykernel install --user --name pixel-racers --display-name "pixel-racers"
   git lfs install && git lfs pull   # SfM maps + track-model mcap
   ```

2. **Point the notebook at your mcap** — open [`submission-demo/run_submission.ipynb`](submission-demo/run_submission.ipynb), edit the **User configuration** cell:

   | Setting | Eval (training + GT) | Predict (test, no GT) |
   |---------|----------------------|------------------------|
   | `MODE` | `"eval"` | `"predict"` |
   | Mcap path | `TRAIN_MCAP = "/absolute/path/to/training_0.mcap"` | `TEST_MCAP = "/absolute/path/to/test_0.mcap"` |
   | Laps | `LAPS = [5, 6, 7]` or `None` | N/A (full bag unless you set windows) |
   | Initial pose | ignored (GT used) | `INIT_POSE_XYYAW = (x, y, yaw)` |

   Use **absolute paths** to `.mcap` files on your machine. You do **not** need a pre-built `extracted/` directory; assets are extracted on first SETUP run.

3. **Run cells in order:** User config → SETUP → Option 1 → Option 2 → Option 3 → Summary → Deliverables.

Full step-by-step tables, examples, and output layout: **[`submission-demo/README.md`](submission-demo/README.md)**.

Example submission artifacts: [`submission-files/`](submission-files/).

---

## 1. Environment

Same as quick start above. Use Jupyter kernel **`pixel-racers`**. On Apple Silicon, SfM notebooks expect **MPS**.

Optional real HF-Net ONNX: [`3d-map-test/models/README_HFNET.md`](3d-map-test/models/README_HFNET.md). The submission demo defaults to **`sp_nv_nn`** (SuperPoint + NetVLAD + mutual NN).

### Git LFS (submission maps)

`submission-demo/sfm-model/` (`map_bundle.npz`, `netvlad_db.npz`, `points3D.ply`) and `submission-demo/track-model/map.mcap` are stored with **Git LFS** (see `.gitattributes`).

```bash
brew install git-lfs
git lfs install
git clone git@github.com:baihao/PixelRacers.git
cd PixelRacers
git lfs pull
git lfs ls-files
```

After `git pull`, run `git lfs pull` if map files look like short pointer text.

**Maintainers — push code first, LFS second:**

```bash
GIT_LFS_SKIP_PUSH=1 git push origin main   # code + pointers only
git lfs push origin main                   # ~1GB maps (retryable)
```

---

## 2. Data layout (outside the repo)

Typical sibling directories (not committed):

```text
SchubKI/
  c0_2026-10-13_22-53-42_camloc/
    c0_2026-10-13_22-53-42_camloc_0.mcap
    extracted/          # optional; submission-demo extracts its own under outputs/
  rosbag_..._cameras_lap/
    ..._0.mcap          # test bag for predict mode
  PixelRacers/          # this repo
```

Lap time windows: [`lapdata/laps.json`](lapdata/laps.json).

Heavy experiment outputs under `outputs/` / `outputs_*` are gitignored.

---

## 3. Components

| Path | Role |
|------|------|
| [`submission-demo/`](submission-demo/) | **Submission pipeline** — `run_submission.ipynb`, bundled maps, three options |
| [`submission-files/`](submission-files/) | Example EKF submission + evaluate report |
| `camloc_play.py` | Read camera JPEG from mcap |
| `lapdata/` | Lap `start_ns` / `end_ns` |
| `odometry/` | SE dead-reckoning baseline |
| `state-error-estimation/` | SE step residuals → process noise MADσ |
| `3d-map-test/` | Lap2+4 mapping + Lap3 multi-cam loc + σ(s) |
| `query-demo/` | Lap3 SE vs EKF (three-camera sync + se2) |
| `fr-auto-mask/polygons.json` | `camera_fr` ego mask |
| `third_party/Hierarchical-Localization` | hloc submodule |

---

## 4. Recommended workflow (research notebooks)

### 4.1 Play mcap

```bash
source pixel-racers/bin/activate
python camloc_play.py --list-topics
python camloc_play.py --camera fl --first
```

Notebook: `test/play_mcap.ipynb`.

### 4.2 Velocity odometry (optional)

`odometry/lap2_velocity_odometry.ipynb` — integrate SE v/a/ω vs GT at camera times.

### 4.3 Process noise MADσ (for EKF)

`state-error-estimation/se_step_residual_noise.ipynb` → `se_step_residual_noise.json`. A copy ships in `submission-demo/error-model/`.

### 4.4 SfM + Lap3 localization

`3d-map-test/sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb` — see `3d-map-test/SFM_MULTI_LAP_S_ORDER.md`.

### 4.5 EKF query-demo

`query-demo/lap3_ekf_query_demo.ipynb`, `query-demo/lap5_ekf_query_demo.ipynb` — see `query-demo/QUERY_DEMO_EKF.md`.

Official evaluate (per-lap GT crop):

```bash
python submission-demo/tools/evaluate.py \
  query-demo/outputs/lap3/gt_lap3.csv \
  query-demo/outputs/lap3/ekf_cam/submission.csv
```

---

## 5. FAQ

| Issue | Fix |
|-------|-----|
| Missing `lightglue` / deps | Activate venv; reinstall `requirements.txt`; init submodules |
| OpenMP duplicate library | `export KMP_DUPLICATE_LIB_OK=TRUE` |
| evaluate coverage &lt; 90% | Use lap-cropped GT, not full-trip `gt.csv` |
| Maps are tiny text files | `git lfs pull` at repo root |
| query-demo missing loc / σ(s) | Run 3d-map-test notebook through fused σ(s) export |

---

## 6. Submodule

```bash
git submodule update --init --recursive
```

Upstream: [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization).
