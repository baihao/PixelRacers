# camera_r: Lap2+Lap4 joint mapping by track arc length \(s\) + Lap3 localization (design)

> **Status:** **Notebook implemented** — `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb`  
> Helper API: `sfm_hfnet.py` (`require_mps` / `project_frames_sn` / `build_map_s_neighbors` / …)  
> Baseline: `sfm_hfnet_lap2_map_lap3_loc.ipynb` (single-lap temporal neighborhood ±2).  
> \(s,n\): `23-00-38_cut_120m_update/tools` (`track_frame.py`).

---

## 1. Goals

| Item | Choice |
|---|---|
| Cameras | **`r` / `fr` / `rl`** (separate map+query each; `fr` uses ego mask) |
| Feature / localization stack | **HF-Net** (`sfm_hfnet.py`: prefer `hfnet_onnx`, else proxy `sp_nv_nn`; matching is mutual NN, **no LightGlue**) |
| Acceleration | **MPS required** (Apple Silicon; `sfm.pick_device()` → `mps`; fail explicitly if unavailable, no silent CPU fallback) |
| Mapping laps | **Lap 2 + Lap 4** (keyframes from both laps in one SfM / NetVLAD database) |
| Query lap | **Lap 3** (not used for mapping, avoids train/test leakage) |
| Neighborhood definition | No longer “time axis ±k”; use **index neighborhood after sorting by track arc length \(s\)** |
| Match window | For sorted frame \(i\): match \(i\pm1,\;i\pm2,\;i\pm3\) (at most 6 edges / frame) |
| Evaluation | Same as existing notebook: **Option A/B metrics + error curves + trajectory plots** |

**Motivation:** With single-lap temporal ±2 matching, cross-lap co-visible geometry at the same track location is unused; if Lap2 and Lap4 are merged but neighbors are still temporal, Lap2 end and Lap4 start are not adjacent in time. Project frames onto racing-line \(s\) and sort so **frames on the same corner/straight across laps become neighbors**, thickening 3D points in previously sparse failure regions.

---

## 2. Why \(x,y \rightarrow s,n\) then sort by \(s\)

When joint-mapping two (or more) laps:

1. Use the **racing line** in `map.mcap` (`/constructor0/map_ar/racing_line`) to project each frame’s GT \((x,y[,yaw])\) to \((s,n[,epsi])\).  
   - Implementation: `track_frame.Line.xy_to_sn` / `convert.py xy2sn`  
   - Path: `23-00-38_cut_120m_update/map.mcap` (lap length ≈ **2976.875 m**)
2. \(s \in [0, L)\), \(L=\) one-lap arc length; cross-lap frames’ \(s\) lie on the same period and are directly comparable.  
3. **Stable sort** all mapping frames from **Lap2∪Lap4** by ascending \(s\) (tie-break on `stamp_ns` if equal \(s\)).  
4. On ordered sequence \(\{F_0,\ldots,F_{N-1}\}\), for each \(i\) match with  
   \[
   j \in \{i-3,i-2,i-1,i+1,i+2,i+3\} \bmod N
   \]  
   then GT epipolar triangulation (loop closure: frames near start/finish connect to the other end via mod \(N\)—for robustness, **circular arc-length distance** \(s\)-neighbors instead of pure index neighborhood is an option; see §5).

> Note: Wording like “if lap 2 and 3 are mapped together” describes the **same multi-lap merge logic**; this design **actually uses Lap2+Lap4 for mapping and Lap3 for query** so the query lap never enters the map.

---

## 3. Pipeline (planned)

```text
Lap2 camera_r ──┐
                ├─→ undistort + GT pose + (x,y,yaw)→(s,n)
Lap4 camera_r ──┘
                ↓
         sort by s (loop closure)
                ↓
    for i vs i±1,±2,±3: local descriptor mutual NN match (no LightGlue)
                ↓
         GT epipolar + triangulation → points3D
                ↓
         global descriptor database (all mapping frames)
                ↓
Lap3 camera_r query → top-5 → mutual NN → PnP (MPS throughout)
                ↓
     metrics (A/B) + err_vs_time + traj_vs_gt
```

> **Matcher clarification:** This design matches `sfm_hfnet.py`—local matching is **mutual NN + ratio test**, **not** SuperGlue / LightGlue. The LightGlue path in legacy `sfm_experiment` is out of scope for this experiment.

### 3.1 HF-Net + MPS (hard requirements)

The HF-Net paper uses **one network** for local features + global descriptors; the repo has two backends (see `models/README_HFNET.md`):

| `BACKEND` | What it is | Matching |
|---|---|---|
| **`hfnet_onnx`** | **True HF-Net** (`models/hfnet.onnx`, joint local+global) | mutual NN |
| **`sp_nv_nn`** | **Proxy stack**: separate SuperPoint + NetVLAD, same pipeline shape as the paper, **not** monolithic MobileNet | mutual NN |

| Item | Requirement |
|---|---|
| Module | Reuse `sfm_hfnet.py` + existing HF-Net notebook mapping / localize / metrics path |
| Preferred backend | **`hfnet_onnx`** when `models/hfnet.onnx` exists; else **`sp_nv_nn`** temporarily, mark `proxy=true` in `run_meta.json` |
| Forbidden | **LightGlue / SuperGlue** (not matchers for this experiment; do not conflate with “HF-Net”) |
| Device | Notebook startup: `device = sfm.pick_device()`, **assert `device == "mps"`**; features and globals on MPS |
| Performance intent | More laps × wider window → rely on MPS + NN (vs LightGlue) for wall clock; OpenCV PnP / triangulation may stay on CPU |
| Baseline | Compare to single-lap `outputs_hfnet/map_r` under **same backend + same MPS** |

Suggested output root: `outputs_hfnet_multi_lap_s/` (alongside `outputs_hfnet/`).

---

## 4. Differences vs single-lap temporal neighborhood

| | Current (`sfm_hfnet_*`) | This design |
|---|---|---|
| Feature stack | HF-Net (onnx first / else `sp_nv_nn` proxy) | **Same; matching = NN, not LightGlue** |
| Device | MPS (proven) | **Same: MPS required** |
| Mapping data | Single lap (e.g. Lap2) | Lap2 + Lap4 |
| Frame order | Timestamps | **Racing-line \(s\)** |
| Match neighborhood | Time \(i\pm1,\pm2\) | \(s\)-order \(i\pm1,\pm2,\pm3\) |
| Same location cross-lap | No | Can neighbor and triangulate |
| Query | Lap3 | Lap3 (same) |
| Evaluation | A/B + trajectory | Same, vs “Lap2-only mapping” baseline |

---

## 5. Implementation details (review points)

### 5.1 \(s,n\) projection

- Input: per frame `gt_x, gt_y, gt_yaw, stamp_ns`  
- Output: `s, n, epsi` (racing line); optional pit line, but mapping neighborhood **uses track \(s\) only**  
- Batch projection: `xy_to_sn(..., t_ns=stamps)` with local window acceleration (same as on-vehicle)

### 5.2 Sorting and neighborhood

**Option A (index neighborhood, simple)**  
After sort by \(s\), take \(i\pm1,\pm2,\pm3\) (mod \(N\)).  
Risk: uneven sampling density across laps → index ±3 may correspond to very different \(\Delta s\).

**Option B (arc-length neighborhood, recommended in code)**  
For each \(i\), under circular distance  
\[
d_L(s_i,s_j)=\min\bigl(|s_i-s_j|,\; L-|s_i-s_j|\bigr)
\]  
take the nearest \(K=6\) other frames, or up to 6 frames with \(d_L < \tau_s\) (e.g. 15–25 m).  
Closer to “~3 samples ahead/behind along the track.”

Review preference: **implement B as primary, A as ablation.** The title “i±1..±3” is default post-sort behavior; if sampling is roughly uniform, A ≈ B.

### 5.3 Geometry gates (reuse / tune)

- Minimum baseline \(\|t\|\): cross-lap same \(s\), different \(n\) (racing line) → baseline may be mostly lateral, still triangulable; skip if too close (e.g. \(\|t\|<0.5\,\mathrm{m}\))  
- Epipolar threshold, ambiguity filter, PnP `min_inl`: align with existing hfnet experiments for fair comparison  
- Same physical lamp may appear twice from two laps: acceptable duplicate points, or later voxel merge (not v1)

### 5.4 NetVLAD database

- Database = global descriptors for all Lap2∪Lap4 keyframes  
- Lap3 query top-5: may hit “same corner, other lap” frames—expected

### 5.5 Output layout (planned)

```text
3d-map-test/outputs_hfnet_multi_lap_s/
  map_r_lap2+4/          # map_bundle, points3D, netvlad, frames_with_s.csv
  loc_lap3_r/            # errors.csv, submission_xy.csv, plots
  metrics_official.csv
  run_meta.json          # backend, device=mps, stride, neighbor_mode
```

`frames_with_s.csv`: `stamp_ns, lap, s, n, epsi, x, y, yaw` to sanity-check sort.  
`run_meta.json`: record `BACKEND`, `device=mps`, neighborhood mode for reproducible comparison with single-lap HF-Net runs.

### 5.6 MPS usage convention

- Startup cell: print and verify `sfm.pick_device()` → `mps`; on failure `raise RuntimeError` (do not run full multi-lap experiment on slow CPU pretending success).  
- SP / NetVLAD / (if used) HF-Net ONNX Runtime EP: prefer GPU/MPS path; same loading as existing `sfm_hfnet`.  
- After batch feature cache to disk, localization phase also uses MPS—avoid “mapping MPS, query CPU” half acceleration.

### 5.7 Evaluation (same as existing notebook)

For successful Lap3 poses:

| Metric | Source |
|---|---|
| Option A Position / lat / lon RMSE, Yaw RMSE | map \(x,y\) |
| Option B \(s,n\) RMSE | same `map.mcap` + `track_frame` |
| within 1.0 m [%] | — |
| Runtime / frame | localize timing |
| Plots | `err_vs_time.png`, `traj_vs_gt.png` |

Baseline: **same query (Lap3) + same hyperparameters**, “Lap2-only temporal neighborhood map” vs “Lap2+4 \(s\)-neighborhood map”.

---

## 6. Laps and data (`laps.json`)

| Lap | Role | duration | path |
|---|---|---|---|
| 2 | Mapping | ~167.6 s | ~2993 m |
| 3 | **Query** | ~123.6 s | ~3024 m |
| 4 | Mapping | ~101.3 s | ~2979 m |

Frame counts (stride=1, ~10 Hz): Lap2≈1600–1700, Lap4≈1000–1100 → mapping total ≈ **2.6k–2.8k** frames; \(s\)-order match edges are several× the temporal scheme—watch runtime (smoke test with `MAP_STRIDE=2` first).

---

## 7. Risks

| Risk | Description | Mitigation |
|---|---|---|
| \(s\) projection error | Large lateral offset / pit → large \(n\), \(s\) still usable but neighborhood semantics degrade | Optional: drop mapping frames with \(\|n\|>n_{\max}\) |
| Loop mod arithmetic | Start/finish index neighbors may span large gaps | Prefer circular arc-length neighbors |
| Duplicate / conflicting observations | Small appearance differences across laps → mismatches | Epipolar + reprojection; ambiguity filter |
| Compute | frames × window width ↑ | **MPS required** + stride + at most 6 edges per frame |
| MPS unavailable | Non-Apple / driver issues | Hard fail with message; no default full run on CPU |
| Query leakage | Accidentally adding Lap3 to map inflates results | **Hard rule: mapping laps 2+4 only** |

---

## 8. Suggested implementation order (after review)

1. New notebook (suggested name `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb`): first cell checks **MPS + HF-Net backend**.  
2. Smoke `frames_with_s.csv`: Lap2+4 projection, sort, visualize \(s\) histogram / color along track.  
3. Implement `build_map_s_neighbors(...)` (arc-length neighbors + optional index ±3), features via `sfm_hfnet` / MPS.  
4. Full or stride=2 build `map_r_lap2+4` → `outputs_hfnet_multi_lap_s/`.  
5. Lap3 query + metrics/trajectory (reuse `compute_camloc_metrics` / hfnet plotting).  
6. Table comparison vs single-lap `outputs_hfnet/map_r` (same backend, same MPS).

---

## 9. Review checklist (please confirm)

- [x] **HF-Net required** (prefer `hfnet_onnx`; `sp_nv_nn` proxy only without weights)  
- [x] **Local matching = mutual NN** (explicitly **no LightGlue**)  
- [x] **MPS required** (startup check; mapping + query features on MPS)  
- [ ] Mapping laps confirmed **Lap2 + Lap4** (not 2+3)  
- [ ] Query confirmed **Lap3 only**  
- [ ] Neighborhood: **sorted i±1..±3**, and/or **circular arc-length Top-6**  
- [ ] v1: drop frames with \(\|n\|>n_{\max}\)?  
- [ ] Smoke stride / full-lap strategy  

Entry point: `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb` → `outputs_hfnet_multi_lap_s/`.
