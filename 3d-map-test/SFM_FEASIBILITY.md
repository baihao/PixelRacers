# camera_r / camera_fl: Lap2 mapping + Lap3 retrieval localization feasibility

> Directory: `PixelRacers/3d-map-test`  
> Primary data: `/constructor0/sensor/camera_r/compressed_image` + `camera_info`  
> Extension: `/constructor0/sensor/camera_fl/compressed_image` + `camera_info` (**same mapping idea as `r`**, separate map)  
> Mapping: `laps.json` **lap 2**　｜　Localization eval: `laps.json` **lap 3**  
> This document is feasibility analysis only; `camera_r` is validated in a notebook, `camera_fl` is a parallel extension.

---

## 1. Goals (what you are building)

### Phase A — Mapping (Lap 2)

For rear camera `camera_r`:

1. SuperPoint local features  
2. **Descriptor ambiguity** filtering  
3. **LightGlue** match with **2 frames before/after**  
4. **GT pose** + **epipolar / RANSAC** match filtering  
5. Triangulate sparse **SfM 3D point map**  
6. **NetVLAD** global descriptors for retrieval database  

### Phase B — Localization eval (Lap 3, after mapping)

For each query frame (or subsample) on lap 3, `camera_r`:

1. SuperPoint detection  
2. Descriptor ambiguity filtering  
3. NetVLAD global descriptor → retrieve **top-5** from Lap2 database  
4. **LightGlue** between query and each of 5 candidates  
5. Localize with SfM map (2D–3D + **PnP / RANSAC**), compare error to Lap3 GT  

Given: images and timestamps, intrinsics `K,D`, extrinsics `tf_static`, GT (for dev eval).

| Asset (Lap2 output) | Lap3 use |
|---|---|
| NetVLAD `{g_i}` + frame index | Retrieve 5 candidates |
| Sparse 3D points + per-frame observations | After LightGlue, 2D–3D for PnP |
| Candidate pose \(T_{\mathrm{map}\leftarrow\mathrm{cam}}\) | Initial guess / weak fallback |

### Extension — `camera_fl` (parallel to `r`, independent map)

Front/rear overlap is minimal—**do not share 3D points or NetVLAD DB**. For `camera_fl`, repeat Phase A/B (see **§10**):

1. Lap2: undistort `fl` → SuperPoint → ambiguity filter → ±2 LightGlue → GT epipolar triangulation → NetVLAD DB  
2. Lap3: same preprocessing → top-5 → LightGlue → SfM 2D–3D PnP → GT error  
3. **No fusion first:** separate `loc_lap3_r` / `loc_lap3_fl`, time-align for aggregate metrics and complementary failures (§10.4); fusion only after clear complementarity (§10.5)

---

## 2. Problem reframing: is this classic SfM?

| | Classic SfM (COLMAP, etc.) | This approach |
|---|---|---|
| Pose | Unknown, from matches | **Known (GT)** |
| Scale | Often ambiguous | **Metric (GT)** |
| Main difficulty | Pose drift, loop closure | Match quality, extrinsic error, night repeated texture |
| In essence | Structure-from-Motion | **Known-pose multi-view triangulation / mapping** |

Conclusion: with GT available, **feasibility is higher** than blind SfM.  
More accurate name: **GT-posed sparse reconstruction** (SfM-style feature–match–triangulate pipeline).

Task docs allow GT in development; for final pure-visual submission, this map acts as a **prior map**, consistent with “no GT at eval time.”

---

## 3. Data and geometry

### 3.1 Time and frame counts

| | Lap 2 (map) | Lap 3 (query) |
|---|---|---|
| `start_ns` | `1791918444832466282` | `1791918612426651869` |
| `end_ns` | `1791918612426651869` | `1791918736054678423` |
| Duration | ≈ **167.6 s** | ≈ **123.6 s** |
| Path length | ≈ 2993 m | ≈ 3024 m |
| Approx frames @10 Hz | ≈ **1600–1700** | ≈ **1200–1300** |

Lap2 ±2 edges \(O(4N)≈7\times10^3\); Lap3 full lap ×5 LightGlue ≈ \(6\times10^3\) matches—manageable.  
Similar path, same rear camera → **revisitable appearance**, good cross-lap localization test (closer to deployment than same-lap self-retrieval).

### 3.2 Camera `r` (vs fl/fr)

From dataset README Accuracy notes:

- **Pitch**: reliable all cameras  
- **Yaw**: reliable for **`r` and `rl`** (fl/fr ≈ ±1°)  
- **Roll / translation**: not calibrated (mount values)

`camera_r` mount: roughly rear of car (`t_x≈-2.28 m`), axis ~horizontal, rearward.  
Vs forward stereo:

- More trustworthy heading → **stronger epipolar constraints**  
- Rear view often sees barriers, lane markings, ground structure—not only beacons  
- Roll and height/longitudinal mount error remain → systematic triangulation bias, usually OK for sparse map

### 3.3 Pose chain (camera in map per frame)

\[
T_{\mathrm{map}\leftarrow\mathrm{cam}}
=
T_{\mathrm{map}\leftarrow\mathrm{base}}(t)
\;
T_{\mathrm{base}\leftarrow\mathrm{cam}}
\]

- \(T_{\mathrm{map}\leftarrow\mathrm{base}}\): GT (`gt.csv` / SE), ~ **0.1 m / 0.2°**  
- \(T_{\mathrm{base}\leftarrow\mathrm{cam}}\): `tf_static`  

Relative pose for frames \(i,j\):

\[
T_{i\leftarrow j}
=
T_{\mathrm{cam}_i\leftarrow\mathrm{map}}
\;
T_{\mathrm{map}\leftarrow\mathrm{cam}_j}
\]

Essential / projection matrices, epipolar geometry, triangulation—**no** `recoverPose`.

### 3.4 Motion baseline (scale)

Adjacent frames \(\Delta t≈0.1\,\mathrm{s}\); lap 2 speed ~15–20 m/s:

| Match span | Approx baseline |
|---|---|
| ±1 frame | ≈ 1.5–2.0 m |
| ±2 frames | ≈ 3–4 m |

Much larger than fl–fr baseline (≈0.36 m)—**better depth geometry**.  
Low speed / stop segments shorten baseline → depth variance grows; drop by \(\|t\|\) or disparity threshold.

---

## 4. Proposed pipeline (aligned with requirements)

```text
lap2 camera_r images
    ↓
undistort (K,D)
    ├────────────────────────────────────────────┐
    ↓                                            ↓
SuperPoint local features                      NetVLAD global g_i
    ↓                                            ↓
descriptor ambiguity filter                    L2 normalize + store (HDF5/NPZ)
    ↓                                            ↓
±2 LightGlue + GT epipolar/RANSAC              frame index ↔ stamp ↔ T_map_cam
    ↓                                            (for later top-K retrieval)
triangulation → sparse points in map frame
```

Local mapping and global descriptors **share the same undistorted images and timestamps**—one pass, parallel extraction, consistent preprocessing.

### 4.1 SuperPoint

- Input: undistorted `camera_r` (optional resize, e.g. long edge 512–1024)  
- Output: per frame \(K_i, D_i\) (descriptors)  
- Risk: night lights still cluster → filtering required next

### 4.2 Descriptor ambiguity filter (concrete recipe)

For descriptors \(\{d_k\}\) on one image:

1. For each \(k\), nearest neighbor \(k'\) **within the same image** (exclude self)  
2. If \(\|d_k-d_{k'}\| < \tau_{\mathrm{dup}}\) → drop or downweight  
3. Or keep uniqueness score \(\|d_k-d_{k'}\|\), keep top quantile  

Optional: bright lamp mask, grid quota (max N points per cell).  
Filter **before LightGlue** to keep ambiguous points out of the matcher.

### 4.3 Temporal window ±2 + LightGlue

- **Matcher: LightGlue** (`features="superpoint"`), SuperPoint official pairing; repo `lightglue` / hloc  
- Input: filtered keypoints + descriptors; output: matches + scores  
- Optional score cutoff (e.g. > 0.1) before epipolar / RANSAC  
- Small window → small appearance change, high overlap, better match rate than arbitrary pairs  
- Pure forward/rear motion: far points low parallax; near (barriers) more useful  
- Bare NN not main path (ablation only)

### 4.4 GT + epipolar / RANSAC

With known \(R,t\) from GT:

\[
E = [t]_\times R,\quad
F = K^{-T} E K^{-1}
\]

Filtering:

1. **Epipolar distance** \(|\mathbf{x}'^\top F\mathbf{x}| / \|\cdot\| < \tau_{\mathrm{px}}\) (e.g. 1–2 px)  
2. Optional RANSAC on inliers to estimate \(E\) (diagnose GT/extrinsic consistency)  
3. After triangulation: **positive depth both sides**, reprojection < \(\tau_{\mathrm{reproj}}\), depth ∈ \([z_{\min}, z_{\max}]\)

More stable than blind RANSAC pose + triangulation, matches “GT known” setup.

### 4.5 Build 3D model

- Each accepted match → two-view triangulation \(X\) (unify to map frame)  
- Multi-frame observations: descriptor track or pixel neighborhood → fuse (mean/median), count observations  
- Output:  
  - `points3D` (xyz + optional color)  
  - `observations` (point_id, frame_id, uv)  
  - optional COLMAP / Open3D / PLY export  

### 4.6 NetVLAD global descriptors (retrieval localization)

**Purpose:** At query time, avoid matching whole map locally—use global vector \(g_q\) to find **top-K similar keyframes**, then refine with local features / associated 3D points.

**Build database (mapping, lap2 `camera_r`)**

1. NetVLAD on each undistorted \(I_i\) (hloc `netvlad` / Pitts30k weights)  
2. \(g_i \in \mathbb{R}^{D}\) (often \(D=4096\), PCA to 256/512 optional)  
3. L2 normalize: \(g_i \leftarrow g_i / \|g_i\|\)  
4. Store per row:

| Field | Meaning |
|---|---|
| `frame_id` / `stamp_ns` | Align with local features, GT |
| `g_i` | Global descriptor |
| `T_map_cam` or `xyzyaw` | Pose prior after retrieval hit |
| `image_path` | Optional debug |

Can use repo **hloc** (`third_party/Hierarchical-Localization`) `extract_features` + NetVLAD → HDF5, COLMAP/hloc retrieval compatible.

**Night track notes**

- Lamp arrays → **perceptual aliasing**: need **top-5 + geometric verify**, not top-1 alone  
- DB and query both `camera_r`, same undistortion  
- Optional: Lap2 keyframe subsample by arc length to reduce redundant neighbors  

---

## 5. Phase B: Lap3 localization (after mapping)

```text
Lap3 query frame q (camera_r)
    ↓
undistort (same K,D as mapping)
    ↓
┌──────────────────────┬──────────────────────┐
↓                      ↓
SuperPoint             NetVLAD → g_q
↓                      ↓
ambiguity filter       cosine/dot vs Lap2 {g_i}
↓                      ↓
local (k_q, d_q)       top-5 candidates {c1…c5}
└──────────┬───────────┘
           ↓
for each candidate c_k:
    LightGlue(q, c_k)
    → matched pixels → via c_k obs table to 2D–3D (X in map)
    → optional epipolar/reproj prescreen
           ↓
merge 2D–3D from 5 candidates (dedupe point_id)
           ↓
solvePnPRansac (K_r) → T_cam←map or T_map←cam
           ↓
to base_link: T_map_base = T_map_cam · inv(T_base_cam)
           ↓
vs Lap3 GT: err_xy / err_yaw (eval only)
```

### 5.1 Query preprocessing (same as mapping)

1. **SuperPoint** (same resize etc. as Lap2)  
2. **Ambiguity filter** (same thresholds)  
3. **NetVLAD** \(g_q\), L2 normalize  

### 5.2 Retrieval: fixed top-5

- Similarity: \(s_i = g_q^\top g_i\) (cosine if normalized)  
- **K = 5** highest Lap2 keyframes  
- Offline **Recall@5** with GT (hit: at least one candidate within 5 m or 10 m of query GT, yaw < 30°)

### 5.3 LightGlue + SfM localization

For each candidate \(c_k\):

1. LightGlue \((u_q, u_{c})\)  
2. SfM obs: if \(u_c\) maps to point \(X\), query **2D–3D** \((u_q, X)\)  
3. Too few 3D: fallback 2D–2D + known \(T_{c_k}\) chain to map (secondary)  

After merging 5 candidates:

- `cv2.solvePnPRansac` (or SQPnP)  
- Success gates: inliers, reprojection (e.g. inl ≥ 12, med reproj < 3 px)  
- Failure → lost / weak fallback to best candidate pose (track separately in eval)

**Note:** Lap3 GT **not** used in estimation—GT for scoring only; consistent with “no GT at inference” (first-frame init can be separate).

### 5.4 Cross-lap challenges (Lap2 map → Lap3 query)

| Issue | Description | Mitigation |
|---|---|---|
| Trajectories not identical | Lateral offset, speed | top-5; relax hit distance |
| Lighting / exposure | Same night usually similar | Same preprocessing |
| Sparse map | Some candidates lack 3D | Merge multi-candidate correspondences; ablate filter strength |
| Perceptual aliasing | All 5 on wrong straight | PnP inlier gate; failure rate stats |

---

## 6. Feasibility assessment

### 6.1 Overall

| Dimension | Verdict | Notes |
|---|---|---|
| Lap2 mapping geometry | **High** | GT pose + meter-scale temporal baseline |
| Calibration usability | **Med–high** | `r` yaw/pitch favorable; roll/t mount values |
| Lap2 local matching | **Medium** | Night lamps; ambiguity + epipolar |
| Lap3 retrieval R@5 | **Med–high** | Same camera cross-lap revisit; K=5 + PnP verify |
| Lap3 SfM localization | **Medium (worth doing)** | Map density + retrieval on correct segment |
| Engineering scale | **Manageable** | ~7k mapping edges; ~1.2k×5 LightGlue localize |

**Overall: Lap2 mapping + Lap3 top-5 retrieval localization is feasible—recommended main validation path.**

### 6.2 Main risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Repeated lamps → wrong matches | Mapping outliers / loc mismatch | Ambiguity filter + highlight mask + epipolar |
| Roll / mount translation error | Cloud / PnP bias | Monitor reprojection; optional extrinsic refine |
| NetVLAD aliasing | top-5 all wrong segment | PnP inlier gate; failure stats |
| Cross-lap lateral offset | Less co-visibility | Merge 5-candidate 2D–3D; relax eval distance |
| Sparse map | PnP fail | Widen depth range; observation count threshold |
| Low-speed short baseline | Poor depth | Skip if \(\|t\|<\tau_t\) |

### 6.3 vs “true SfM (unknown pose)”

- No pose optimization in mapping: simple, metric scale.  
- Localization uses **map points + PnP**, not full SfM again.  
- Phase 1: custom triangulation + PLY + retrieval loc is enough.

---

## 7. Success criteria (suggested)

### 7.1 Lap2 mapping

1. Median inlier ratio after epipolar > 30% (or > 50 inliers/edge)  
2. Median triangulation reprojection < 1–2 px; depth sane (e.g. 2–80 m)  
3. Point cloud along track corridor, no obvious outlier clouds  

### 7.2 Lap3 retrieval (K=5)

1. **Recall@5**: at least one candidate within 5 m (or 10 m) of query GT, yaw < 30°; target R@5 > 0.7–0.8 if possible (measure)  
2. Log R@1 but not sole metric  

### 7.3 Lap3 SfM localization

1. **Success rate:** fraction of queries passing PnP gates (e.g. inl≥12)  
2. **Error** (successes): median `err_xy`, median `|err_yaw|`; baseline vs SE integration  
3. **Failure analysis:** retrieval vs insufficient matches vs PnP blow-up  

---

## 8. Suggested implementation order

**Mapping (Lap2)**

1. Sample 20–50 frames: SuperPoint → ambiguity viz  
2. NetVLAD smoke (neighbors high sim, far low)  
3. \(i,i+1\) epipolar + triangulation  
4. ±2 window short trajectory PLY  
5. Full Lap2: `points3D` + `global_descriptors.h5` + `frames.csv` + obs table  

**Localization (Lap3)**

6. Extract Lap3 images (or stride)  
7. Per frame: SuperPoint → filter → NetVLAD → **top-5**  
8. LightGlue on 5 candidates → 2D–3D → PnP  
9. vs Lap3 GT: `err_xy` / `err_yaw` curves and tables  
10. (Optional) `rl`, multi-lap map, keyframe ablation  
11. **Extend `camera_fl`:** same scripts/hyperparams, swap topic and `T_base_fl` / `K_fl` (§10)  
12. (Optional) Lap3 dual-camera pose fusion eval (§10.4)  

Suggested tree:

```text
3d-map-test/
  SFM_FEASIBILITY.md
  data/lap2_r/                 # mapping images (camera_r)
  data/lap3_r/                 # query images (camera_r)
  data/lap2_fl/                # mapping images (camera_fl)
  data/lap3_fl/                 # query images (camera_fl)
  outputs/
    map_r/                     # points3D, observations, frames, netvlad (r)
    map_fl/                    # same, independent (fl)
    loc_lap3_r/                # top5, poses, errors.csv
    loc_lap3_fl/
    loc_lap3_compare/          # aligned compare: union success, complementary fails (first)
    loc_lap3_fused/            # optional: after complementarity confirmed
  scripts/
```

---

## 9. Conclusions

- **Lap2:** `camera_r` + GT sparse map + NetVLAD DB — **feasible** (full-lap experiment exists).  
- **Lap3:** same camera query → SuperPoint → filter → NetVLAD → **5 candidates** → LightGlue → **SfM 2D–3D PnP** — **feasible, full closed-loop validation**.  
- **Extension:** `camera_fl` **same mapping recipe**, Lap2 independent map, Lap3 independent query — **feasible**; **no shared features/point cloud** with `r`.  
- **Eval strategy:** **no fusion first**—align compare success/error and **at-least-one-success** union (§10.4); fuse poses only after clear complementarity (§10.5).  
- Lap3 GT not fed to estimator, scoring only.  
- Main risks: night aliasing and map density; `r` failures often sparse local map + PnP reject, not NetVLAD—forward view may cover different texture.  
- Deliverables: `map_r` + `loc_lap3_r`; `map_fl` + `loc_lap3_fl`; **dual-path alignment table** (union success); fusion report optional, later.

---

## 10. Extension: `camera_fl` Lap2 map + Lap3 query

> **Algorithm-isomorphic** to §1–§5 `camera_r`; swap stream, intrinsics, extrinsics, output dirs.  
> Front/rear almost no feature overlap → **independent SfM / NetVLAD**; never match `fl` features into `map_r` (or vice versa).

### 10.1 Why add forward view, why separate maps

| Point | Description |
|---|---|
| Complementary FOV | `r` rear, `fl` forward-left; failures often correlate with rear texture / map holes—forward may add landmarks |
| No co-visibility | Front/rear feature intersection ≈ empty → no basis for joint triangulation or shared descriptor DB |
| Unified pose frame | Both maps’ 3D in same `map` frame (\(T_{\mathrm{map}\leftarrow\mathrm{cam}}=T_{\mathrm{map}\leftarrow\mathrm{base}}\,T_{\mathrm{base}\leftarrow\mathrm{cam}}\)) |
| Calibration | README: `fl`/`fr` **yaw ~ ±1°** (`r`/`rl` stabler); pitch still reliable. Slightly looser epipolar than `r` (Appendix B, start 3–6 px) |

`camera_fl` mount: forward-left (`t≈(0.94, 0.20, 0.34)` m) vs `r` rear (`t_x≈-2.28` m).

### 10.2 Phase A' — Lap2 build `Map_fl` (same recipe)

Pipeline identical to §4, input Lap2 `camera_fl`:

```text
lap2 camera_fl images
    ↓
undistort (K_fl, D_fl)
    ├────────────────────────────┐
    ↓                            ↓
SuperPoint + ambiguity filter    NetVLAD → {g_i^fl}
    ↓                            ↓
±2 LightGlue + GT epipolar       store netvlad_db_fl
    ↓
triangulation → points3D_fl (map frame)
```

Implementation notes:

1. **Time window / lap:** same as `r`, `laps.json` lap 2 `[start_ns, end_ns]`.  
2. **Pose chain:** \(T_{\mathrm{base}\leftarrow\mathrm{fl}}\) from `tf_static` `base_link→camera_fl`; GT still `base_link` xyzyaw.  
3. **Hyperparams:** align with validated `r` smoke/full (e.g. `SP_RESIZE=512`, `SP_MAX_KP`, `MIN_NN_DIST`, `MATCH_WINDOW=±2`, `EPI_THRESH_PX`); if epipolar inliers low, increase `EPI_THRESH_PX` first (yaw noise).  
4. **Output:** `outputs/map_fl/` (`points3D.ply`, observations, `netvlad_db.npz`, frames)—**physically separate** from `map_r`.  
5. **Success:** §7.1 + compare point density along arc vs `r` for complementarity.

### 10.3 Phase B' — Lap3 query `Map_fl` with `fl`

Isomorphic to §5, query stream Lap3 `camera_fl` only:

```text
Lap3 q_fl
  → undistort / SuperPoint / filter / NetVLAD
  → top-5 on Map_fl {g_i^fl} only
  → LightGlue(q_fl, c_k^fl) → 2D–3D (points3D_fl only)
  → solvePnPRansac(K_fl) → T_map_cam_fl
  → T_map_base = T_map_cam_fl · inv(T_base_fl)
  → vs Lap3 GT err_xy / err_yaw
```

Constraints:

- Retrieval, candidates, 3D obs **all from `Map_fl`**.  
- No Lap3 GT in estimate; GT for scoring only.  
- Same fields as `r` (`stamp_ns`, `ok`, `n_pnp_corr`, `n_pnp_inl`, `err_xy`, `err_yaw`) for §10.4 alignment.  
- Success §7.2–§7.3; **no pose fusion** in this phase.

### 10.4 No fusion first: dual-path compare and complementarity (current priority)

Goal: see if `fl` is usable overall and **rescues `r` failures** before investing in fusion.

**Time alignment** (`r`/`fl` header stamps differ ~µs, motion negligible):

1. Lap3 timeline; for each `r` query frame nearest `fl` (or reverse; median \(|\Delta t|\) ≪ 1 ms).  
2. Per instant \((\mathrm{ok}_r,\mathrm{ok}_{fl},\mathrm{err}_r,\mathrm{err}_{fl})\).  
3. Write `outputs/loc_lap3_compare/pair_errors.csv`.

**Aggregate metrics (one row each camera)**

| Metric | `camera_r` | `camera_fl` |
|---|---|---|
| Success rate \(\mathrm{ok}\) | measured ~ **57%** (709/1235) | TBD |
| Median `err_xy` / p95 (successes) | existing `errors.csv` | TBD |
| Median \|err_yaw\| | same | TBD |
| Recall@5 (optional) | same | TBD |

**Complementarity (“can one path fix the other’s failure?”)**

Let \(A=\mathrm{ok}_r\), \(B=\mathrm{ok}_{fl}\) (aligned booleans):

| Quantity | Definition | Interpretation |
|---|---|---|
| \(P(A)\) | `r` success rate | Rear baseline |
| \(P(B)\) | `fl` success rate | Forward standalone |
| \(P(A\cup B)\) | **At least one success** | If ≫ \(\max(P(A),P(B))\) → complementary |
| \(P(B\mid \neg A)\) | **`fl` success when `r` failed** | Direct “rescue” measure |
| \(P(A\mid \neg B)\) | `r` success when `fl` failed | Symmetric |
| \(P(\neg A\land \neg B)\) | Dual failure rate | Hard regions even fusion won’t fix |
| r-only / fl-only / both | counts and fractions | Plot on trajectory: do failures separate? |

Heuristic decision (task-dependent):

- If \(P(B\mid\neg A) \gtrsim 0.3\) and \(P(A\cup B)-P(A) \gtrsim 0.1\) → **worth fusion / selection later**;  
- If \(P(A\cup B)\approx P(A)\) (`fl` failures overlap `r`) → forward helps little on failures; thicken map first;  
- If \(P(B)\) << `r` but high \(P(B\mid\neg A)\) → spare tire on failure; hard selection fusion OK.

**Visualization (suggested)**

1. Trajectory: `r` fail, `fl` fail, dual fail colors;  
2. Timeline: both `ok` strips + union strip;  
3. For `r` `fail_regions`, per region report \(P(B\mid\neg A)\).

**`camera_r` baseline (Lap3 full lap)**

- Hard failure ~ **43%** (526/1235), mainly sparse local map + PnP reject; NetVLAD rarely root cause.  
- Failure regions often tens–hundreds of consecutive frames → if `fl` has high \(P(B\mid\neg A)\) there, dual-camera value is clear.

### 10.5 Multi-camera pose fusion (`r` / `fr` / `rl`)

Independent SfM → `base_link` poses; fuse at **pose layer** (not features, not merged 2D–3D):

\[
\omega = \mathbf{1}[\mathrm{ok}] \cdot n_{\mathrm{inl}} \cdot
\exp\!\big(-\bar{e}_{\mathrm{reproj}} / \sigma\big)
\quad(\sigma\sim 4\,\mathrm{px})
\]

Without reprojection error, \(\omega=\mathbf{1}[\mathrm{ok}]\cdot n_{\mathrm{inl}}\).

| Mode | Method |
|---|---|
| Hard select | Max \(\omega\) |
| SE(2) weighted | \(\alpha_c=\omega_c/\sum\omega\); linear \(x,y\), circular yaw mean |
| Gating | Drop camera if >5 m or >15° from highest-weight pose |

Timestamps: `r`/`fr`/`rl` differ ~µs, nearest-neighbor align.  
Code/eval: `3d-map-test/sfm_hfnet.py` (`fuse_multi_camera_tables`) + notebook `sfm_hfnet_lap2_map_lap3_loc.ipynb` §4.

### 10.6 Feasibility and risks (vs `r`)

| Dimension | Verdict | Notes |
|---|---|---|
| Algorithm reuse | **High** | Same code path, swap camera |
| Geometry / GT | **High** | Same GT-posed triangulation; meter baselines |
| Calibration | **Medium** | yaw ±1° → epipolar/triangulation slightly worse than `r` |
| Appearance | **Med–high** | Different forward texture; **complementarity from §10.4 measurement** |
| Engineering | **~×2** | Second map+loc pass; compare script light |

**Overall: finish `Map_fl` + independent Lap3 query, use §10.4 union/conditional success for complementarity; fusion not current goal.**

### 10.7 Suggested order (`fl`, no fusion)

1. Parameterize `sfm_experiment` with `camera_name ∈ {r, fl}` (topic, `K,D`, `T_base_cam`, prefix).  
2. Lap2 short smoke `fl` (features → ±2 triangulation → PLY).  
3. Full Lap2 `map_fl` + NetVLAD.  
4. Full Lap3 loc → `loc_lap3_fl/errors.csv` (fields match `loc_lap3_r`).  
5. **Aligned compare (§10.4):** \(P(A),P(B),P(A\cup B),P(B\mid\neg A)\) + trajectory/failure plots → fusion decision.  
6. (Only if 5 positive) implement §10.5.

---

## Appendix A: Key data paths

| Resource | Path |
|---|---|
| mcap | `c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap` |
| topic `r` | `/constructor0/sensor/camera_r/compressed_image` |
| topic `fl` | `/constructor0/sensor/camera_fl/compressed_image` |
| intrinsics | `extracted/camera_info.json` → `r` / `fl` |
| extrinsics | `extracted/tf_static.json` → `base_link→camera_r` / `base_link→camera_fl` |
| GT | `extracted/gt.csv` (`base_link` in `map`) |
| mapping lap | `laps.json` → **lap 2** |
| query lap | `laps.json` → **lap 3** |

## Appendix B: Epipolar threshold intuition

Focal length \(f≈1491\,\mathrm{px}\). Relative pose error ~ \(0.2°\) (GT scale):

\[
\delta \approx f \cdot \tan(0.2^\circ) \approx 5\,\mathrm{px}
\]

With extrinsic error stacked, start epipolar threshold **2–3 px** and tune; too tight kills true matches, too loose admits lamp mismatches. Joint tune with inlier rate and reprojection.

For **`camera_fl`**: yaw calibration ~ ±1° → upper-bound intuition \(\delta \approx f\cdot\tan(1^\circ)\approx 26\,\mathrm{px}\); in practice GT+mount error usually below that—start **3–6 px** (same order as `r` experiments at 6 px), refine by inlier rate.
