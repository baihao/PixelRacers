# PixelRacers

Constructor Labs track **camera localization**: mcap playback, HF-Net / sp_nv_nn SfM mapping & localization, state-estimation process noise, Lap3 EKF query-demo, and a **self-contained submission pipeline** in [`submission-demo/`](submission-demo/).

---

## Quick start — test the submission pipeline

The fastest way to verify the repo is to run **`submission-demo/run_submission.ipynb`** on a training or test `.mcap`.

1. **Environment** (once):

   ```bash
   cd PixelRacers
   bash scripts/setup_pixel_racers_venv.sh    # creates ./pixel-racers, submodules, pip install, Jupyter kernel
   source scripts/env.sh
   git lfs install && git lfs pull              # optional: submission SfM maps + track-model mcap
   ```

   Options: `--recreate` (delete old venv), `--no-kernel`, `--skip-submodule`. See [`scripts/setup_pixel_racers_venv.sh`](scripts/setup_pixel_racers_venv.sh).

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

### Reference performance (two-camera localize, Option 2 / 3)

Measured on **MacBook Air, Apple M3, 2024** with the default **`sp_nv_nn`** backend (`run_submission.ipynb`, `QUERY_STRIDE = 1`).

**Per camera (fr and rl separately)** — one query frame:

- **~100 ms per image** for the vision localize step: CLAHE/preprocess → SuperPoint + NetVLAD features → NetVLAD retrieval → mutual-NN match → PnP (timings also appear in the eval table as `loc_ms_mean_fr` / `loc_ms_mean_rl`).

**End-to-end pipeline (why overall ≈ 2 FPS)** — Options 2 and 3 are built on **two-camera** operation, not a single stream:

1. **Data processing** — read JPEG from `.mcap`, decode, undistort, ego mask, sync **fr** and **rl** to logical frames (`t_last`).
2. **Two localize passes** — fr and rl each run the ~100 ms/image path (often sequentially on one GPU); both must finish before fusion.
3. **Option 2 (two-camera)** — nearest-stamp alignment and **SE(2) fusion** of fr+rl poses into one submission stamp.
4. **Option 3 (EKF)** — same vision cost as above, plus SE prediction, extrapolation of both cameras to `t_last`, fused SE(2) measurement, and **EKF update** per sync bucket.

Counting mcap I/O, dual-camera localize, and fusion/EKF overhead together, **overall throughput is about 2 FPS** on that hardware (effective logical frames per second at full stride, both cameras). Option 1 (motion only, no vision) is much faster.

Second runs hit **`loc/.../errors.csv` cache** and skip step 2; times improve accordingly. Use a larger `QUERY_STRIDE` for smoke tests.

---

## 1. Environment

**Recommended:** run the setup script from the repo root:

```bash
cd PixelRacers
bash scripts/setup_pixel_racers_venv.sh
source scripts/env.sh
```

This creates the virtualenv at **`pixel-racers/`** (gitignored), initializes **`third_party/Hierarchical-Localization`**, installs [`requirements.txt`](requirements.txt), and registers the Jupyter kernel **`pixel-racers`**.

| Step | Manual equivalent |
|------|-------------------|
| Venv | `python3.12 -m venv pixel-racers && source pixel-racers/bin/activate` |
| Submodule | `git submodule update --init --recursive` |
| Deps | `python -m pip install -U pip && pip install -r requirements.txt` |
| Kernel | `python -m ipykernel install --user --name pixel-racers --display-name "pixel-racers"` |

On Apple Silicon, SfM notebooks expect **MPS**. Optional HF-Net ONNX: [`3d-map-test/models/README_HFNET.md`](3d-map-test/models/README_HFNET.md). Submission demo default backend: **`sp_nv_nn`** (SuperPoint + NetVLAD + mutual NN).

### Third-party software and models

PixelRacers depends on **hloc**, **LightGlue / SuperPoint**, **PyTorch**, **NetVLAD (Pitts30K weights)**, **OpenCV**, **mcap**, and other PyPI packages. Weights and licenses are summarized in **[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)** (standard third-party notices file). Review it before redistribution or publication.

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
