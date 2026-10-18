# PixelRacers Camera Localization — 5-Page PPT Content Brief

> Purpose: Finalize copy, formulas, figures, and data sources per slide in this MD first, then build the PPT.  
> Data sources: `paper/1812.03506v2.pdf`, `3d-map-test/`, `odometry/`, `state-error-estimation/`, `query-demo/`.  
> Default eval lap: **Lap3** (mapping on Lap2 / Lap2+4); official EKF scores: **Lap5–7** (fr+rl).

---

## Global Layout Suggestions

| Item | Suggestion |
|---|---|
| Page count | **5 pages** (principles / camera experiments / motion & error / EKF / improvements) |
| Style | Dark track atmosphere or light engineering theme; each slide **1 main title + 1 tagline**, avoid card stacks |
| Fonts | Distinct sans-serif for titles; render formulas in LaTeX and paste as images |
| Figures | Prefer existing repo PNGs (paths under each slide’s “Suggested figures”) |

---

# Page 1 | Camera Stream Localization Algorithm

## Title
**Hierarchical Visual Localization: Global Retrieval to 6-DoF Pose**

Subtitle: HF-Net hierarchical localization (Sarlin et al., arXiv:1812.03506) and this project’s per-camera SfM map implementation

## Story to tell (talking points)

1. **Problem**: From onboard multi-camera JPEG streams (~10 Hz), estimate `base_link` \((x,y,\psi)\) in `map`.  
2. **Classic approach (paper)**: Coarse-to-fine hierarchical localization, not dense match against the full database.  
3. **Vehicle constraints**: Under calibration and mounting, **stereo is not viable**; use **per-camera independent SfM maps + online query**.

## Block A — Paper (1812.03506)

Paper title: **From Coarse to Fine: Robust Hierarchical Localization at Large Scale** (HF-Net).

**Pipeline (for slides):**

```text
Query image
  → (1) Global descriptor (NetVLAD / global feature) → Top-K retrieval in map keyframe DB
  → (2) Local feature detect & describe (SuperPoint / HF-Net local)
  → (3) Mutual NN match with candidate keyframes
  → (4) 2D–3D correspondences + PnP (+ RANSAC) → camera pose
  → (5) Extrinsic transform to base_link → (x, y, yaw)
```

**Why hierarchical:**  
At large scale, full-database local matching is infeasible; global retrieval shrinks search to a few keyframes, then precise geometry locally—robustness and efficiency.

**Mapping to PixelRacers (`3d-map-test/sfm_hfnet*.py` / notebook):**

| Paper step | PixelRacers implementation |
|---|---|
| Global retrieval | NetVLAD top-K (neighborhood may use racing-line \(s\) order / arc neighborhood) |
| Local features | HF-Net ONNX or `sp_nv_nn` (MPS) |
| Matching | mutual NN + ratio test |
| Geometry | OpenCV PnP + inlier stats |
| Map | Offline SfM / triangulation on selected laps, cache `map_{cam}_lap*/map_bundle.npz` |

## Block B — Why Not Stereo; Per-Camera Independent SfM

**Calibration and mounting facts (PPT bullets):**

- Among seven cameras, **`fl, fr, r, rl` are relatively trustworthy** (other side cameras have larger calibration bias).  
- Even for these: **intrinsics, orientation, focal length differ strongly** (e.g. `fl`/`fr` narrow FOV focal length gap), **stereo baseline and extrinsics do not support stable stereo depth / stereo VO**.  
- `fl` and `fr` overlap heavily, but `fl` is narrower FOV → fewer detectable features.

**Architecture adopted:**

```text
Offline (per camera):
  image sequence_c + GT pose  →  SfM / 3D landmarks  →  map_c

Online (per camera query):
  current frame_c  →  hierarchical localize on map_c  →  pose_c
  (multi-camera SE(2) fusion / EKF on pages 2 & 4)
```

One-liner: **Do not treat multi-camera as stereo pairs; each stream is a “monocular localization sensor with its own map”.**

## Suggested figures

| Position | Content |
|---|---|
| Top-left | Paper-style Coarse→Fine flow (custom) |
| Top-right | Vehicle camera layout (mark fl/fr/r/rl) |
| Bottom | “Independent map_c” three-column: fr / r / rl query + trajectory snippet each |

## Footer sources
`paper/1812.03506v2.pdf` · `PixelRacers/3d-map-test/`

---

# Page 2 | Localization Analysis and Camera Selection

## Title
**Which Cameras? Single-Lap vs Dual-Lap Mapping? How to Fuse?**

Subtitle: Lap3 query · map Lap2 vs Lap2+4 · single camera vs se2 fusion

## Block A — Camera selection logic

1. Candidates: `fl, fr, r, rl` (relatively reliable calibration).  
2. **Drop `fl`**: highly similar scene to `fr`, but narrower FOV, fewer features → poor cost/benefit.  
3. **Experiment trio: `fr, r, rl`** (same as `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb`).

## Block B — Mapping strategy (Lap2-only vs Lap2+4)

Eval: **same Lap3 query**.  
Compare runs: `lap2_time` (lap 2 only) vs `lap2+4_s` (laps 2+4, \(s\)-ordered mapping).

**Official-style metrics (position RMSE on successful frames; success rate in parentheses):**

| Camera | Lap2-only A-RMSE [m] | success | Lap2+4 + s-order A-RMSE [m] | success |
|---|---:|---:|---:|---:|
| **r** | 7.10 | 42.3% | **2.89** | **69.4%** |
| **fr** | 7.65 | 72.9% | **4.16** | **90.4%** |
| **rl** | 18.59 | 56.7% | **7.42** | **72.2%** |

**Conclusion (under figure caption):** Multi-lap map + racing-line \(s\) neighborhood cuts RMSE and raises success; thereafter always **`map_*_lap2+4`**.

## Block C — Three-camera fusion (brief)

After alignment to the same query timeline:

1. Each stream yields \(\hat{z}_c=(x_c,y_c,\psi_c)\) and confidence weight  
   \(\omega_c \propto \mathbf{1}[\mathrm{ok}_c]\cdot n_{\mathrm{inl},c}\cdot e^{-\bar{e}_{\mathrm{reproj}}/\sigma}\).  
2. **`fuse_se2_weighted`**: SE(2) weighted fusion by \(\omega\); gate \(\Delta xy\le 5\,\mathrm{m}\), \(\Delta\psi\le 15^\circ\).  
3. Baseline: **hard** = take max-\(\omega\) camera only.

## Block D — Single camera vs fusion (Lap3, map=lap2+4)

| Method | success | pos RMSE [m] | within 1 m (successful frames) | within 1 m (all query frames) |
|---|---:|---:|---:|---:|
| r | 69.4% | 2.89 | 58.3% | 40.5% |
| fr | 90.4% | 4.16 | 72.2% | 65.3% |
| rl | 72.2% | 7.42 | 81.0% | 58.5% |
| **fused_se2** | **96.4%** | **4.42** | **75.4%** | **72.7%** |
| fused_hard | 96.4% | 4.45 | 73.7% | 71.1% |

**Interpretation:**  
- Fusion lifts **coverage/success** to ~96%; overall within-1m (all frames) clearly beats any single stream.  
- RMSE sits between best and worst single camera: fusion prioritizes “more often solved + outlier suppression”, not only minimizing RMSE on already-successful frames.

## Suggested figures (trajectories required)

| Figure | Path |
|---|---|
| Single-camera trajectories | `3d-map-test/outputs_hfnet_multi_lap_s/loc_lap3_{r,fr,rl}/traj_vs_gt.png` |
| Fused trajectory | `.../loc_lap3_fused/traj_fused_se2.png` |
| Lap2 vs Lap2+4 bar chart | notebook export compare / redraw from `metrics_official_compare.csv` |
| Single vs fusion bar chart | `.../loc_lap3_fused/metrics_fuse_compare.png` |
| Error vs distance | `.../loc_lap3_fused/err_vs_s_fused_se2.png` |

## Footer sources
`sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb` · `outputs_hfnet_multi_lap_s/`

---

# Page 3 | Motion Prediction Error × Image Localization Error (\(s\))

## Title
**Two Uncertainty Sources: Dynamic Dead Reckoning vs Visual Localization**

Subtitle: Why visual noise must be calibrated experimentally — feature detect/match error is hard to model analytically

## Block A — Dynamic position prediction (see `odometry/`)

**Input:** `state_estimation` \(v_x,v_y,a_x,a_y,\omega_z\) (**do not** use SE \(x,y\) as observations; first-frame pose may initialize).

**Integration (body → map):**

\[
\Delta \mathbf{p}_{\mathrm{body}} = \mathbf{v}\,\Delta t + \tfrac12 \mathbf{a}\,\Delta t^2,\quad
\psi \leftarrow \psi + \omega_z\Delta t
\]

Rotate displacement to `map` with mid-interval heading; **reset velocity** with measured \(\mathbf{v}\) each step to avoid \(a\) double-integration blow-up.

**Experimental scale (Lap2 dead reckoning sampled at FL stamp, `lap2_velocity_odometry.ipynb`):**

| Mode | xy median [m] | xy p95 [m] | endpoint error [m] |
|---|---:|---:|---:|
| \(v\) only | ≈ 8.37 | ≈ 22.9 | ≈ 15.7 |
| \(v+a\) | ≈ 8.26 | ≈ 23.0 | ≈ 15.7 |

**Conclusion:** Pure dead reckoning drifts to **several to ~10+ meters** over a lap; vision correction is required; dead reckoning suits **high-rate prediction**, not standalone final localization.

## Block B — Motion step noise (see `state-error-estimation/`)

Residuals between SE reports and pose finite differences; robust stats:

\[
\mathrm{MAD}\sigma = 1.4826 \cdot \mathrm{median}\big(|r-\mathrm{median}(r)|\big)
\]

EKF process noise anchors (full lap):

| Channel | MADσ | Unit |
|---|---:|---|
| \(\sigma_{v_x}\) | 0.073 | m/s |
| \(\sigma_{v_y}\) | 0.020 | m/s |
| \(\sigma_{a_x}\) | 10.59 | m/s² |
| \(\sigma_{a_y}\) | 8.63 | m/s² |
| \(\sigma_{\omega_z}\) | 0.0031 | rad/s |

Suggested figures: `state-error-estimation/se_step_residuals_vs_gaussian.png`, `se_step_residuals_qq.png`.

## Block C — Image localization error vs track distance \(s\) (multi-lap notebook)

On racing-line arc length \(s\), statistics on fused/single-stream error; fit **\(\sigma_{\mathrm{obs}}(s)\)** (multi-Gaussian / bin-MAD, then sharpen) for EKF observation covariance.

Suggested figures:

- `loc_lap3_fused/obs_error_sigma_vs_s.png`  
- `loc_lap3_fused/sigma_multigauss_vs_s.png`  
- `loc_lap3_fused/err_vs_s_fused_se2.png`

## Block D — Why experimental image localization error?

Prominent callout (user intent, engineering wording):

> Visual error comes **not only** from mounting/intrinsic bias and imaging noise, but mainly from:  
> **feature detection error, descriptor noise, cross-frame matching mistakes, PnP outliers** — nonlinear and scene-dependent, **hard to quantify with closed-form priors**.  
> Therefore **Lap3 query × GT** empirically estimates \(\sigma(s)\) along \(s\), then feeds the filter.

## Block E — Work to reduce detect / matching error (bullets)

- HF-Net / SuperPoint-class local features + NetVLAD global retrieval (hierarchy shrinks false-match space)  
- mutual NN + ratio test; PnP RANSAC + inlier count in fusion weights  
- **`fr` ego-mask** (ego vehicle polygon), reduce hood/body false features  
- Multi-lap map (Lap2+4) and **\(s\)-neighborhood keyframe selection**, reduce appearance ambiguity  
- Fusion gates (position/heading consistency) suppress single-stream blow-ups  

## Footer sources
`odometry/lap2_velocity_odometry.ipynb` · `state-error-estimation/se_step_residual_noise.ipynb` · `loc_lap3_fused/obs_error_sigma_*`

---

# Page 4 | EKF: Motion Prediction + Image Correction

## Title
**Extended Kalman Filter: High-Rate Dead Reckoning × Asynchronous Vision Updates**

Subtitle: Noise from page 3; flow and formulas in `QUERY_DEMO_EKF.md`; results from Lap5–7 fr+rl demo

## Block A — System roles

| Input | Role |
|---|---|
| SE: \(v,a,\omega_z\) @ ~106 Hz | **Predict** |
| Cameras fr / r / rl @ ~10 Hz (final experiment **fr+rl**) | **Localize → Fuse → Update** |
| First-frame SE pose | **Init only** |
| GT / SE pose sequence | **Not runtime observations** |

> **Camera choice (consistent with page 2):** Final EKF uses **`fr` + `rl`** (not `fl`).  
> vs three cameras: one fewer saves compute; vs `r+rl`: `fr+rl` balances accuracy–latency on Lap3/5 (see query-demo two-camera comparison).

## Block B — Full math flow (half-slide formula block)

**State:**

\[
\mathbf{x}=[x,\;y,\;\psi]^\top
\]

**1. Predict (motion model \(f\), control \(u=(v_x,v_y,a_x,a_y,\omega_z)\))**

\[
\mathbf{x}_{k|k-1}=f(\mathbf{x}_{k-1|k-1},u_k,\Delta t),\quad
P_{k|k-1}=F_k P_{k-1|k-1} F_k^\top + Q_k
\]

\[
Q \approx G\,\mathrm{diag}(\sigma_{v_x}^2,\sigma_{v_y}^2,\sigma_{a_x}^2,\sigma_{a_y}^2,\sigma_{\omega}^2)\,G^\top
\]

(\(\sigma_\bullet\) = page 3 MADσ)

**2. Async cameras → align to \(t_{\mathrm{last}}\)**

\[
t_{\mathrm{last}}=\max_c t_c,\quad
\hat{z}_c(t_{\mathrm{last}})=\mathrm{Propagate}\big(\hat{z}_c(t_c),\,u,\,t_c\to t_{\mathrm{last}}\big)
\]

**3. Fused observation**

\[
z=\mathrm{fuse\_se2}\big(\{\hat{z}_c(t_{\mathrm{last}})\},\{\omega_c\}\big)
\]

**4. Update (observation noise vs distance)**

\[
\hat{s}=\mathrm{xy\_to\_s}(\mathbf{x}_{k|k-1}),\quad
R(\hat{s})=\mathrm{diag}\big(\sigma_{\mathrm{obs}}^2(\hat{s}),\,\sigma_{\mathrm{obs}}^2(\hat{s}),\,\sigma_\psi^2\big)
\]

\[
K=P H^\top(HPH^\top+R)^{-1},\quad
\mathbf{x}\leftarrow\mathbf{x}+K(z-h(\mathbf{x})),\quad
P\leftarrow(I-KH)P
\]

\(h(\mathbf{x})=\mathbf{x}\); wrap yaw innovation to \((-\pi,\pi]\). On vision failure **predict only, no update** (preserve coverage).

**Flow sketch:**

```text
SE(v,a,ω) ──predict──► x,P
                         │
Camera frames ──localize──► z_c ──propagate──► fuse_se2 ──update(R(s))──► x,P
```

## Block C — Experimental results (official `evaluate.py`)

Method: **ekf_cam_2cam (fr+rl)** · map Lap2+4 · eval Lap5 / 6 / 7  
submissions: `query-demo/outputs/lap5_7_2cam_fr_rl/submissions/`

| Window | poses | coverage | **pos RMSE [m]** | lat / lon RMSE | yaw RMSE [deg] | within 1 m |
|---|---:|---:|---:|---|---:|---:|
| Lap5 | 1523 | 100% | **1.116** | 0.849 / 0.724 | 0.445 | 73.3% |
| Lap6 | 1205 | 100% | **0.875** | 0.260 / 0.835 | 0.215 | 81.9% |
| Lap7 | 1095 | 100% | **0.867** | 0.385 / 0.777 | 0.204 | 77.4% |
| Lap5–7 pooled | 3823 | 100% | **0.976** | 0.592 / 0.776 | 0.325 | 77.2% |

**Baseline (same setup, indicative, Lap5):**  
Pure SE dead reckoning RMSE ~11 m; offline fr+rl fusion ~1.8 m; **EKF ~1.1 m** — balance of accuracy and online feasibility.

## Suggested figures

| Figure | Path |
|---|---|
| Trajectories | `query-demo/outputs/lap5_7_2cam_fr_rl/lap{5,6,7}/compare/traj_vs_gt.png` |
| Error vs time | `.../compare/err_vs_time.png` |
| Mode bar chart | `.../compare/metrics_official.png` |
| Cross-lap summary | `.../metrics_cross_lap.png` |
| Beside formulas | EKF predict/update block diagram (custom) |

## Footer sources
`query-demo/QUERY_DEMO_EKF.md` · `lap5_7_ekf_2cam_fr_rl_query_demo.ipynb` · `submissions/evaluate_metrics.txt`

---

# Page 5 | Improvement Directions

## Title
**Next Steps: More Robust Features, Better Maps, Stronger Lateral Constraints, Broader Appearance Coverage**

## Improvements (four items, 2×2 or four vertical sections)

### 1. Other Vehicles / Dynamic-Object Mask
**Vehicle detection** before feature extraction; mask other cars to avoid dynamic features in the map, reducing localization jitter and blow-ups.

### 2. Maps: Mapping Laps + Multi-Lap Optimization (Not Just Stacking Keyframes)
Current **Lap2+Lap4 SfM**. Alternative:  
- Still use those laps to bound map size;  
- Use **Lap1 / Lap3 / Lap5** etc. for **bundle adjustment / landmark refinement**.  
→ **Map stays small, accuracy uses multi-lap training data fully.**

### 3. Lane Lines + High-Precision 2D Track Map
Lane / curb detection with existing `track_bounds` high-precision 2D map to tighten **lateral offset (\(n\))** and further reduce cross-track error.

### 4. Appearance Domain Extension (Daytime)
Training/mapping is almost all **night**. Add **daytime** mapping or domain adaptation for cross-time stability.

## Closing line
The system already achieves **~0.9–1.1 m RMSE, 100% coverage** on multi-lap official metrics; the four items target **robustness, map accuracy, lateral accuracy, generalization**.

## Suggested figures
- Left: failure case screenshots (other car in frame / night overexposure)  
- Right: improvement roadmap (Mask → Map BA → Lane → Day)

---

## Appendix: PPT Production Checklist

- [ ] Page 1: paper pipeline + “per-camera independent SfM” architecture  
- [ ] Page 2: **must include** `traj_vs_gt` / `traj_fused_se2`, plus Lap2 vs Lap2+4 and fused tables  
- [ ] Page 3: dead-reckoning error curves + MADσ histogram + \(\sigma(s)\) curve  
- [ ] Page 4: EKF formula block + Lap5–7 trajectories + official evaluate table (stress **fr+rl**, not fl)  
- [ ] Page 5: four improvements, one icon each  

### Key Path Quick Reference

```text
paper/1812.03506v2.pdf
PixelRacers/3d-map-test/sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb
PixelRacers/3d-map-test/outputs_hfnet_multi_lap_s/loc_lap3_{r,fr,rl,fused}/
PixelRacers/odometry/lap2_velocity_odometry.ipynb
PixelRacers/state-error-estimation/se_step_residual_noise.ipynb
PixelRacers/query-demo/QUERY_DEMO_EKF.md
PixelRacers/query-demo/lap5_7_ekf_2cam_fr_rl_query_demo.ipynb
PixelRacers/query-demo/outputs/lap5_7_2cam_fr_rl/submissions/
```
