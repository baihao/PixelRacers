# query-demo: Lap3 EKF Localization Simulation (Design Spec)

> Status: **Design document only, for review; code not yet implemented.**  
> Goal: In simulation, drive localization in **ROS2/mcap arrival order**, compare **pure SE dead reckoning** vs **EKF (dead reckoning + r/fr/rl vision)**, output metrics / trajectory plots / camloc submission, and run through `evaluate.py`.

---

## 1. Goals and Scope

| Item | Choice |
|---|---|
| Data | `c0_2026-10-13_22-53-42_camloc_0.mcap` |
| **Eval / Query lap** | **Lap 3** (same as `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb`) |
| **Map** | Mapping from Lap2 + Lap4: `outputs_hfnet_multi_lap_s/map_{r,fr,rl}_lap2+4` (**no rebuild**, load directly) |
| Motion input | `state_estimation` **vx, vy, ax, ay, wz** (**do not** use SE x,y as observations; **first-frame pose** only for init) |
| Vision input | `camera_r` / `camera_fr` / `camera_rl` (fuse after full set; **no single-camera intermediate steps**) |
| Process noise | **MADσ** from `state-error-estimation/se_step_residual_noise` (see §5.2) |
| Observation noise | \(\sigma_{\mathrm{obs}}(s)\): `obs_error_sigma_vs_s.json` / `obs_error_sigma_s.sigma_of_s` |
| Three-camera fusion | `fuse_se2_weighted` (same as notebook; optional hard baseline) |
| Comparison | **A: pure SE prediction** vs **B: EKF prediction + three-camera observation correction** |
| Outputs | Two submissions + metrics + trajectory/error plots + persisted simulation intermediates |

Simulation semantics: **replay along the mcap message timeline**, triggering processing when a given topic arrives, approximating asynchronous arrival on the vehicle.

**Why Lap3 not Lap2:** The SfM map is built from Lap2+Lap4; the notebook’s query/eval lap is Lap3; query-demo reuses the same map and query setup to avoid self-evaluation on the mapping lap.

---

## 2. Motivation: Why EKF + Asynchronous Camera Correction

The offline notebook is “full-lap query → fuse → evaluate”. On the car:

1. **SE (~106 Hz)** continuously provides v/a/ω for prediction;  
2. **Three cameras (~10 Hz)** expose at different times; r / fr / rl timestamps for the same “logical frame” do not match;  
3. Visual innovation must wait until all three are present, and earlier cameras’ poses must be **extrapolated to the last camera’s stamp** before fusion;  
4. Prediction covariance grows over time; observation covariance varies with track position \(s\) (larger in failure regions).

This demo encodes that timing and uncertainty in a unified EKF and compares against a pure dead-reckoning baseline.

---

## 3. Input Topics and Roles

| Topic | Role | Notes |
|---|---|---|
| `/constructor0/state_estimation` | **Prediction driver** | Use `vx,vy,ax,ay,wz`; `x,y,yaw` only for **first-frame init** and eval GT |
| `/constructor0/sensor/camera_{r,fr,rl}/compressed_image` | **Observation** | `header.stamp` = exposure time; submission CSV `timestamp_ns` = last camera stamp of that logical frame |
| `/tf_static`, `camera_info` | Calibration | Load once at startup |
| `/tf` | **Not allowed as input** | Eval / dev reference only |

Lap3 time window: same as `lapdata/laps.json` / multi-lap notebook (`start_ns`…`end_ns`).

---

## 4. System Architecture (Event-Driven)

```text
mcap event stream (Lap3 window, log/header time order)
        │
        ├─ state_estimation @ t_se
        │       → predict(t_prev → t_se)  using v,a,wz
        │
        └─ camera_c image @ t_c   (c ∈ {r,fr,rl})
                → sync_buffer
                → when r,fr,rl all present for same sync key:
                        for each cam:
                            localize(query | map_cam_lap2+4) → z_c @ t_c
                            propagate z_c from t_c → t_last  (v,a,wz)
                        fuse_se2({z_c'}) → z_fuse @ t_last
                        [Mode B only] EKF update(z_fuse, R(ŝ))
                        emit pose @ t_last → submission row
```

### 4.1 Two Run Modes (both required, compared)

| Mode | Name | Behavior |
|---|---|---|
| **A** | `se_predict` | SE dead reckoning only; camera events **do not trigger update** (may still sample output at \(t_{\mathrm{last}}\) for same timeline as B) |
| **B** | `ekf_cam` | SE predict + three-camera sync → extrapolate → se2 fuse → EKF update (\(R(s)\)) |

Same Lap3 window, same initial pose, same sample timestamps for direct comparison.

### 4.2 Sync Key (Same Frame)

Cameras ~10 Hz; three-way stamp skew typically a few ms to tens of ms. First version:

- **Time-window clustering**: anchor on earliest arrival, `|t - t_anchor| < τ_sync` (e.g. 40–80 ms) → same `SyncBucket`;  
- Or **nearest-neighbor pairing**: anchor on `r`, pick nearest stamp for fr/rl (similar to existing `align_by_nearest_stamp`).

Ready when bucket has `r, fr, rl` images; `t_last = max(t_r, t_fr, t_rl)`.

**No single-camera EKF intermediate experiment**; observation path is directly three-camera sync + se2.

---

## 5. Motion Model (see `odometry/`)

Same integration as `lap2_velocity_odometry.ipynb` (applied to Lap3 window), body-frame velocity/acceleration + yaw rate:

\[
\Delta p_{\mathrm{body}} = v\,\Delta t + \tfrac12 a\,\Delta t^2,\quad
\psi \leftarrow \psi + \omega_z\Delta t
\]

Displacements rotated to `map` using mid-interval heading. One integration step per SE interval; **each SE resets velocity state with measured \(v\)** (avoid double integration drift from \(a\)); acceleration mainly enters \(\frac12 a\Delta t^2\) and process noise.

### 5.1 EKF State (Mode B)

\[
\mathbf{x} = [x,\; y,\; \psi]^\top \in SE(2)
\]

First version: \(v,a,\omega\) as **control input \(u\)**. Mode A integrates \(\mathbf{x}\) only, may skip \(P\); Mode B maintains:

\[
\mathbf{x}_{k|k-1} = f(\mathbf{x}_{k-1|k-1},\, u_k,\, \Delta t),\quad
P_{k|k-1} = F_k P_{k-1|k-1} F_k^\top + Q_k(\Delta t, u_k)
\]

### 5.2 Process Noise \(Q\) (SE Step Residual MADσ Directly)

**Do not** calibrate via “pure odometry vs GT”. Process noise from:

- Notebook: `PixelRacers/state-error-estimation/se_step_residual_noise.ipynb`
- Export: `se_step_residual_noise.json` / `se_step_residual_noise_summary.csv`

\[
\mathrm{MAD}\sigma = 1.4826 \cdot \mathrm{median}\big(|r - \mathrm{median}(r)|\big)
\]

Residual meaning: within one SE message, “reported field − reference from pose finite differences”.

**Anchor values written to `config/process_noise.yaml`** (full run, `LAP_FILTER=None`):

| Channel | Residual source | MADσ | Unit | EKF use |
|---|---|---|---|---|
| \(\sigma_{vx}\) | `r_vx` (SE − pose FD) | **0.0729** | m/s | Longitudinal velocity process noise |
| \(\sigma_{vy}\) | `r_vy` | **0.0204** | m/s | Lateral velocity process noise |
| \(\sigma_{ax}\) | `r_ax` (\|v\|>1) | **10.59** | m/s² | Acceleration quadratic-term noise |
| \(\sigma_{ay}\) | `r_ay` | **8.63** | m/s² | Same |
| \(\sigma_{\omega}\) | `r_wz` (`wz` − d(yaw)/dt) | **0.00313** | rad/s | Yaw rate (**use wz, not yaw_vel_rad**) |

Notes:

- Dead reckoning uses **`wz_radps`**; angular rate noise uses `wz_minus_dyaw.mad_sigma`.  
- **Always MADσ**, not sample std.  
- Implementation **reads JSON `mad_sigma`** to generate yaml; table above is documentation anchor only.

\[
Q \approx G\,\mathrm{diag}(\sigma_{vx}^2,\sigma_{vy}^2,\sigma_{ax}^2,\sigma_{ay}^2,\sigma_{\omega}^2)\,G^\top
\]

**Do not use SE pose for process updates**; pose ground truth only for frame-0 init and scoring.

---

## 6. Vision Observation Chain (Reuse Multi-Lap Lap3 Query)

**Map and models: load notebook artifacts directly; no rebuild inside query-demo.**

| Resource | Path (relative to `3d-map-test/`) |
|---|---|
| Map | `outputs_hfnet_multi_lap_s/map_{r,fr,rl}_lap2+4/` (`map_bundle`) |
| Query cache (optional speedup) | notebook `query_meta` / feature cache if present |
| Observation \(\sigma(s)\) | `outputs_hfnet_multi_lap_s/loc_lap3_fused/obs_error_sigma_vs_s.json` |
| Algorithm & hyperparams | `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb` / `sfm_hfnet.run_camera_multi_lap_s_pipeline` localize stage |

For each camera \(c\in\{r,fr,rl\}\) in a synced set:

1. Undistort + (`fr`) ego mask;  
2. HF-Net / `sp_nv_nn` features (MPS);  
3. On `map_{c}_lap2+4`: NetVLAD top-K → mutual NN → PnP;  
4. Obtain \(\hat{z}_c=(x_c,y_c,\psi_c)\) @ stamp \(t_c\), plus \(n_{\mathrm{inl}}\), (optional) reprojection error.

### 6.1 Time Alignment to \(t_{\mathrm{last}}\) (Critical)

Three stamps differ. Before fusion, for each successful observation:

\[
\hat{z}_c(t_{\mathrm{last}})
= \mathrm{Propagate}\big(\hat{z}_c(t_c),\; u(\cdot),\; t_c\to t_{\mathrm{last}}\big)
\]

\(\mathrm{Propagate}\) **reuses the same dead-reckoning model** (SE v/a/ω over \([t_c,t_{\mathrm{last}}]\)).  
Cameras with failed PnP do not enter fusion.

### 6.2 Three-Camera Pose Fusion → Single Observation \(z\)

At \(t_{\mathrm{last}}\):

\[
\omega_c = \mathbf{1}[\mathrm{ok}_c]\cdot n_{\mathrm{inl},c}\cdot\exp(-\bar{e}_{\mathrm{reproj}}/\sigma)
\]

- **se2** (default): `fuse_se2_weighted` (gate \(\Delta xy\le5\,\mathrm{m}\), \(\Delta\psi\le15^\circ\))  
- **hard** (optional baseline): take camera with max \(\omega\)  

Yields \(z=(x,y,\psi)_{t_{\mathrm{last}}}\). **No vision path without this; no single-camera EKF branch.**

---

## 7. EKF Update and \(R(s)\) (Mode B Only)

Project predicted pose \(\mathbf{x}_{k|k-1}\) onto racing line for \(\hat{s}\) (`map.mcap` + `track_frame`).

\[
R(\hat{s}) =
\mathrm{diag}\big(
\sigma_{\mathrm{obs}}^2(\hat{s}),\;
\sigma_{\mathrm{obs}}^2(\hat{s}),\;
\sigma_\psi^2
\big)
\]

\(\sigma_{\mathrm{obs}}(s)\) from multi-Gaussian export; \(\sigma_\psi\) from notebook yaw empirical or constant (e.g. \(1^\circ\sim2^\circ\)).

\[
K = P H^\top (HPH^\top+R)^{-1},\quad
\mathbf{x}\leftarrow\mathbf{x}+K(z-h(\mathbf{x})),\quad
P\leftarrow (I-KH)P
\]

\(h(\mathbf{x})=\mathbf{x}\); wrap yaw innovation to \((-\pi,\pi]\).

Visual fusion failure: **predict only, no update** this cycle; still output predicted pose by default for coverage≥90%.

---

## 8. Simulation Timeline (Pseudocode)

```text
init: load map_{r,fr,rl}_lap2+4, calib, sigma_of_s, Lap3 window
on first SE in window: x ← SE.pose; P ← P0; t ← t_se

for event in mcap_events(Lap3):
  if event is SE:
      predict(t → event.t, u=event.v_a_w)
      t ← event.t
  if event is camera image (r|fr|rl):
      put into sync_buffer
      if buffer ready (r,fr,rl):
          t_last ← max stamps
          predict(t → t_last)
          if mode == ekf_cam:
              for c in {r,fr,rl}:
                  z_c ← localize(c, img_c | map_c_lap2+4) at t_c
                  if ok: z_c ← propagate(z_c, t_c → t_last)
              z ← fuse_se2({z_c})
              if z ok:
                  R ← R_from_sigma(sigma_of_s(s_of(x)))
                  update(z, R)
          append submission (t_last, x, y, yaw)   # Mode A: x is dead reckoning only
          clear buffer
```

Causality: output at time \(t\) uses only images and SE with stamp ≤ \(t\).

---

## 9. Outputs, Comparison, and Validation

### 9.1 Submission CSV (camloc README)

```
timestamp_ns,x,y,yaw
```

- `timestamp_ns`: **last camera** stamp of that logical frame  
- Rate: ~10 Hz; coverage (1 s bins) **≥ 90%**  
- Paths:

```text
query-demo/outputs/lap3/
  gt_lap3.csv                 # clipped GT (evaluate must use this)
  se_predict/submission.csv
  ekf_cam/submission.csv
```

Entry points: `lap3_ekf_query_demo.ipynb` + `query_demo_ekf.py`.  
Vision observations default to reusing `loc_lap3_{r,fr,rl}/errors.csv` (same SfM query, avoid re-running HF-Net in notebook).

### 9.2 Evaluation

```bash
python3 c0_.../tools/evaluate.py outputs/lap3/gt_lap3.csv se_predict/submission.csv
python3 c0_.../tools/evaluate.py outputs/lap3/gt_lap3.csv ekf_cam/submission.csv
```

### 9.3 Required Comparison Artifacts

| Artifact | Description |
|---|---|
| `se_predict/submission.csv` | Mode A |
| `ekf_cam/submission.csv` | Mode B |
| `compare/metrics_compare.csv` | Side-by-side official A/B metrics (RMSE, within 1 m, coverage, etc.) |
| `compare/traj_vs_gt.png` | GT + Mode A + Mode B on one plot |
| `compare/err_vs_time.png` | xy/yaw error vs time for both modes |
| `compare/err_vs_s.png` | Error vs Option B \(s\) for both modes |
| `se_predict/` / `ekf_cam/` each | `metrics_official.csv`, `traj_vs_gt.png`, `err_vs_time.png`, `err_vs_s.png` |
| `ekf_cam/sim/` (optional) | Per frame: `t_last`, per-camera ok/ω, pose before/after fusion, innovation, \(P\) diagonal, \(\sigma(s)\) |

Reuse `compute_camloc_metrics` (same as notebook) for metrics; trajectory/error plot style aligned with `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb`.

---

## 10. Directory Layout (Implemented)

```text
PixelRacers/query-demo/
  QUERY_DEMO_EKF.md
  README.md
  lap3_ekf_query_demo.ipynb  # main entry
  query_demo_ekf.py          # SE/EKF/sync/metrics library
  config/process_noise.yaml
  outputs/lap3/
    gt_lap3.csv
    se_predict/
    ekf_cam/ (+ sim/)
    compare/
```

Reuse:

- `PixelRacers/camloc_play.py` — mcap iteration  
- `PixelRacers/odometry/lap2_velocity_odometry.ipynb` — dead-reckoning formulas (for Lap3)  
- `PixelRacers/state-error-estimation/se_step_residual_noise.json` — process noise MADσ  
- `PixelRacers/3d-map-test/outputs_hfnet_multi_lap_s/map_*_lap2+4` — **existing maps**  
- `PixelRacers/3d-map-test/sfm_hfnet.py` — localize + `fuse_se2_weighted`  
- `PixelRacers/3d-map-test/obs_error_sigma_s.py` — \(R(s)\)  
- `23-00-38_cut_120m_update/tools/track_frame.py` — \(s,n\)

---

## 11. Implementation Order

1. ~~Process noise config~~ → `config/process_noise.yaml`  
2. ~~Mode A pure SE~~ → `outputs/lap3/se_predict/`  
3. ~~Mode B EKF + three cameras~~ → `outputs/lap3/ekf_cam/` (observations from existing `loc_lap3_*`)  
4. ~~Comparison artifacts~~ → `outputs/lap3/compare/`  
5. (Optional) online HF-Net query / full run

**Explicitly out of scope:** single-camera observation EKF intermediate steps.

---

## 12. Risks and Open Items

| Item | Description | Suggestion |
|---|---|---|
| Is SE an “allowed input”? | README forbids pose; **v/a/ω from same SE message** | v1: treat as “IMU/wheel-speed-like”; document boundary |
| Sync window \(\tau_{\mathrm{sync}}\) | Too large mixes frames; too small drops sets | Sweep from 50 ms |
| Extrapolation error | Large \(t_{\mathrm{last}}-t_c\) extrapolation also consumes \(Q\) | Optionally inflate that camera’s observation covariance before fusion |
| Map / query consistency | Map Lap2+4, query Lap3 | Identical to notebook; no map change needed |
| fr mask / MPS | Same as notebook | Hard reuse |
| Real-time | Full HF-Net may be slower than 10 Hz | Demo can run offline in event order; prefer query cache reuse |

---

## 13. Review Checklist

- [x] Eval / Query lap is **Lap 3** (map Lap2+4, load existing models)  
- [x] **No** single-camera EKF; observation path = three-camera sync + extrapolate + se2  
- [x] **Mode A pure SE** first, then **Mode B EKF + cameras**, compare metrics + trajectories and persist  
- [ ] Motion input: SE **v/a/ω**; pose **first frame only**  
- [ ] \(R=\sigma_{\mathrm{obs}}^2(s)\) from multi-Gaussian export  
- [ ] Submission format and `evaluate.py` / coverage≥90%  
- [ ] Process noise: **MADσ** from `se_step_residual_noise` (angular rate uses `wz`)

After sign-off, implement `query-demo` code and run both Lap3 modes per §11.
