# submission-demo 设计文档

> 状态：**设计稿，供 review**。确认后再按本文件生成代码。
> 目标：在 `submission-demo/` 下生成一套**自包含**的、最终用于提交（submission）的定位代码。输入为**原始 `.mcap`**，支持两种运行目标（training 评测 / test 预测）与三种定位方案（motion-only / 两相机定位 / EKF 融合），并内置「在线跑 SfM localize → 结果缓存复用」。
>
> **v2 修订（根据 review）**：
> 1. **标定 / GT / 运动状态不再假设外部 `extracted/` 目录**，一律从输入 `.mcap` 用 `tools/` 脚本提取（`extract_assets.py` → camera_info/tf_static/track_bounds/state_estimation.csv；`extract_gt.py` → gt.csv，仅 training）。
>
> **v3 修订**：确认当前**无 HF-Net 权重**，本版本**退回 `sp_nv_nn`（SuperPoint + NetVLAD + mutual-NN）后端**，与现有 `map_*_lap2+4` bundle 描述子一致，**直接复用现成地图、不重建**。HF-Net 作为未来可选升级（接口 `sfm_features.make_features("hfnet_onnx")` 预留；升级需配套 HF-Net 重建地图 + σ(s) 重标定）。

---

## 1. 需求对齐（来自约定）

| # | 需求 | 设计落点 |
|---|------|---------|
| 1 | setup 可指定 training data path 与 test data path | `SETUP` 配置段：`TRAIN_MCAP` / `TEST_MCAP`（原始 `.mcap`）。**标定/GT/运动状态从 mcap 内用 `tools/` 脚本提取，缓存到 `outputs/<...>/assets/`**，不需要用户另外给 `extracted/` 目录 |
| 2 | setup 可选 **evaluation（基于 training）** 或 **prediction（基于 test）** | `MODE = "eval" | "predict"`；`eval` 出 metrics + submission，`predict` **只出 submission，不算 metrics** |
| 3 | setup 可指定要跑的 lap；不指定则默认全跑 | `LAPS = None`（默认全跑）或 `LAPS = [3]` / `[5,7]`；见 §6 lap 窗口来源与 test 无 GT 的处理 |
| 4 | 三个 option 各对应一个 cell | Option 1 **motion prediction only**；Option 2 **two-camera localization（fr+rl 离线 se2 融合）**；Option 3 **EKF fusion（SE 预测 + fr/rl 外推 → se2 → EKF update，带 R(s)）** |
| 5 | 三个空目录 `sfm-model` / `track-model` / `error-model` 用来放对应模型/资源，做到自包含 | 见 §4 目录与 §5 需要复制/重写的功能清单 |

**自包含原则**：不 `import` `3d-map-test/` 或 `query-demo/` 的模块；把所需功能**重写成 `submission-demo/` 下的 `.py` 文件**（参考已有实现，但不直接搬运 `.ipynb`）。数据资源（地图 bundle、racing line、σ(s) 模型、标定）复制进 `sfm-model/` / `track-model/` / `error-model/`。

---

## 2. 背景：已有流程回顾

参考 `query-demo/lap3_ekf_2cam_fr_rl_query_demo.ipynb` + `query_demo_ekf.py`，整条链路是：

```
原始 mcap ──(每相机抽帧 + 去畸变)──> query frames
   │
   └─ 对 map_{cam}_lap2+4 做 NetVLAD top-K → mutual NN → PnP  ← 在线 SfM localize
         └─ 每相机 errors.csv: stamp_ns, ok, pred_x/y/yaw, n_pnp_inl, (gt_* 仅 eval)
                │
   ┌────────────┴─────────────────────────────────────────────┐
   │                        │                                   │
Option1 motion-only     Option2 两相机 se2 融合            Option3 EKF(2cam)
 SE 航位积分             nearest-stamp ω 加权 SE(2)        SE 预测 + fr/rl 外推到 t_last
 在 t_last 采样          (无 EKF)                           → se2 → EKF update，R(s)=σ²(s)
   │                        │                                   │
   └─────────── 每方案输出 submission.csv (timestamp_ns,x,y,yaw) ─┘
                            │
                    eval 模式: evaluate.py / compute_camloc_metrics
                    predict 模式: 仅落盘 submission.csv
```

与 notebook 的差异（本模块新增）：
- **从 mcap 自提取资源**：notebook 依赖预先 `extract` 好的 `extracted/`；本模块在运行时用 `tools/extract_assets.py` / `tools/extract_gt.py` 从输入 mcap 现提 camera_info / tf_static / track_bounds / state_estimation（+ training 的 gt），缓存到 `outputs/<...>/assets/`。
- **在线 localize（`sp_nv_nn`）**：notebook 直接读预先算好的 `loc_lap3_{fr,rl}/errors.csv`；本模块用 `sp_nv_nn`（SuperPoint+NetVLAD+mutual-NN）对原始 mcap 在线定位，地图复用现成的 `map_{fr,rl}_lap2+4` bundle。
- **缓存复用**：同一组任务参数（mcap 指纹 + 地图指纹 + lap + stride + top_k）跑过一次后，二次运行直接读缓存的 `errors.csv`，不重复跑特征/PnP（复用 `query_cache_valid` + `query_meta.json` 机制）。
- **test 无 GT**：predict 模式不提 `gt.csv`、不算误差、不裁 GT lap。

---

## 3. 运行模式语义

### 3.1 `MODE = "eval"`（training 数据）
- 输入：`TRAIN_MCAP`（原始 mcap）。资源在首跑时**从该 mcap 提取**：`extract_assets.py` → `camera_info.json` / `tf_static.json` / `track_bounds.json` / `state_estimation.csv`；`extract_gt.py` → `gt.csv`（来自 `/tf`）。缓存到 `outputs/eval_<stem>/assets/`。
- lap 窗口：从 `gt.csv` 现算（起终点平面穿越，复用 `lapdata` 逻辑）或读 `lapdata/laps.json`（若同一 mcap）。
- 流程：抽帧时用 `gt.csv` 填 `gt_xyzyaw`（`localize_query` 用它算 `err_xy`，也作评测真值）。
- 产出：三方案各自 `submission.csv` + `metrics_official.csv`（复用 `compute_camloc_metrics`）+ 轨迹/误差图 + `compare/` 汇总。可调用 `tools/evaluate.py` 做官方打分（支持 x/y 与 s/n 两套）。

### 3.2 `MODE = "predict"`（test 数据）
- 输入：`TEST_MCAP`（原始 mcap）。资源：仅 `extract_assets.py` → `camera_info.json` / `tf_static.json` /（可选）`state_estimation.csv`。**不提 gt.csv**（test 无 `/tf` GT）。
- 流程：抽帧时 `gt_xyzyaw` 用占位值 `(nan,nan,nan,nan)`，**不参与**定位结果，只影响被禁用的 err 字段。
- 产出：三方案各自 `submission.csv`（`timestamp_ns,x,y,yaw`），**不**生成 metrics、不裁 GT、不调用 evaluate。

> 关键点 1：`localize_query` 内部算 `err_xy = hypot(pred-gt)`，predict 下 gt 为 nan ⇒ err 为 nan，`ok` 仍由 PnP 内点数决定，不受影响（纯算术，不抛错）。
>
> 关键点 2：`/constructor0/state_estimation`（含 `vx/vy/ax/ay/wz` 与自带 pose）**在 training 与 test 都存在**（它就是车端 SE，不是 `/tf` GT）。所以 Option 1/3 的运动输入在 predict 下**可用**，首帧初值用 SE 首帧 pose。

---

## 4. 目标目录结构

```text
submission-demo/
  DESIGN.md                     # 本文件
  README.md                     # 生成代码后补：如何运行
  run_submission.ipynb          # 主入口 notebook：SETUP cell + 3 个 option cell + 汇总 cell
  config/
    process_noise.yaml          # EKF 过程噪声（由 error-model 的 se_step_residual_noise.json 生成）

  tools/                        # 【已存在】从 mcap 提取资源 + 评测（复用，不重写）
    extract_assets.py           # camera_info/tf_static/track_bounds/state_estimation.csv
    extract_gt.py               # gt.csv（来自 /tf，仅 training）
    convert.py, evaluate.py, track_frame.py

  assets_io.py                  # 封装：调 tools/extract_* 从 mcap 提资源并缓存/加载
  sfm_core.py                   # 自包含 SfM 核心（重写自 sfm_experiment.py 的 localize 子集）
  sfm_features.py               # 特征后端（重写自 sfm_hfnet.py：SPNVJointFeatures 主线 + NNMatcher/mutual_nn_match；HFNetONNX 预留接口）
  sfm_localize.py               # 在线 localize 编排 + 缓存（重写自 run_camera_multi_lap_s_pipeline 的 query 半部）
  mcap_io.py                    # mcap 抽帧（重写自 camloc_play.py：iter_camera_frames/decode_jpeg）
  obs_sigma.py                  # σ(s) 观测噪声模型（重写自 obs_error_sigma_s.py）
  fusion.py                     # se2 融合 + 置信度（重写自 sfm_hfnet.py 的 fuse_* / pose_confidence / align_by_nearest_stamp）
  motion_ekf.py                 # SE 航位积分 + SE2EKF + 外推（重写自 query_demo_ekf.py 的运动/EKF 部分）
  metrics.py                    # compute_camloc_metrics（eval 用；重写自 sfm_hfnet.py）
  pipeline.py                   # 把上面串起来：提资源→(建图)→localize(缓存)→三方案→落盘

  sfm-model/                    # 【资源】sp_nv_nn 预建地图 bundle（自包含，供 localize 加载，不重建）
    map_fr_lap2+4/              # map_bundle.npz, netvlad_db.npz, frames_with_s.csv, points3D.ply（复制现成）
    map_rl_lap2+4/
    models/
      README_HFNET.md           # 未来 HF-Net 升级说明；本版不需要 hfnet.onnx
    polygons.json              # fr/rl ego mask（复制自 fr-auto-mask/polygons.json）

  track-model/                  # 【资源】racing line
    map.mcap                    # 复制自 23-00-38_cut_120m_update/map.mcap（track_frame 读取）

  error-model/                  # 【资源】误差/噪声模型
    obs_error_sigma_vs_s.json   # σ(s)（复制自 loc_lap3_fused/；换 HF-Net 后建议用 HF-Net 结果重标定）
    se_step_residual_noise.json # 过程噪声 MADσ（复制自 state-error-estimation/）

  outputs/                      # 运行产物
    <mode>_<mcap_stem>/
      assets/                   # 从 mcap 提取的 camera_info/tf_static/track_bounds/state_estimation(.csv)/gt.csv(eval)
      loc/loc_<lap>_fr/errors.csv (+ query_meta.json, localize_times_s.npy)
      loc/loc_<lap>_rl/errors.csv ...
      motion_only/submission.csv [+ metrics_official.csv eval]
      two_cam/submission.csv      [+ metrics_official.csv eval]
      ekf/submission.csv          [+ metrics_official.csv eval, sim/ekf_sim.csv]
      compare/metrics_compare.csv, *.png   (eval only)
      gt_lap<lap>.csv             (eval only, 评测用裁剪 GT)
```

> **后端 = `sp_nv_nn`（本版主线）**：`sfm_features` 默认 `SPNVJointFeatures`（SuperPoint 出 local + NetVLAD 出 global），query 阶段用 mutual-NN 匹配（省掉 LightGlue）。与现有 `map_*_lap2+4` 的描述子一致，**直接复用现成地图、不重建**。
>
> **HF-Net 升级预留**：`make_features("hfnet_onnx")` 与 `HFNetONNX` 类保留在 `sfm_features.py`，待拿到 `hfnet.onnx` 后可切；切换时须配套用 HF-Net 重建地图并重标定 σ(s)（见 §14）。

---

## 5. 需要「复制或重写」的功能清单（自包含）

下表列出每个新 `.py` 文件从哪个已有实现**重写**（参考源码逻辑，不搬 notebook），以及对应函数/类。

| 新文件 | 源（参考） | 迁入的函数/类 | 外部依赖 |
|--------|-----------|----------------|----------|
| `assets_io.py` | `tools/extract_assets.py` + `extract_gt.py`（调用，不重写） | `extract_assets(mcap, out_dir)`, `extract_gt(mcap, out)`, `load_assets(dir) -> (camera_info, tf_static, track_bounds, se_df, gt_df?)`；带存在性缓存 | `mcap`, `mcap_ros2` |
| `mcap_io.py` | `camloc_play.py` | `camera_topic`, `stamp_ns_from_msg`, `decode_jpeg`, `ImageFrame`, `iter_camera_frames` | `mcap`, `mcap_ros2`, `cv2`, `numpy` |
| `sfm_core.py` | `sfm_experiment.py` | `pick_device`, `T_from_tf`, `T_map_base`, `T_map_cam`, `pose_xyzyaw`, `relative_T`, `CamCalib`, `load_camera_calib`(从 assets dir), `undistort`, `gt_at`, `FrameData`, `LocalFeatures`(SuperPoint), `NetVLADGlobal`, `filter_ambiguous_descriptors`, `project_points`, `retrieve_topk`, `localize_query`, `MAP_BUNDLE_NAME`, `map_ready`, `load_map_bundle` | `torch`, `lightglue`, `scipy`, `cv2`, `pandas`, `numpy` |
| `sfm_features.py` | `sfm_hfnet.py` | **主线** `SPNVJointFeatures`（SuperPoint+NetVLAD, joint API）、`NNMatcher`、`mutual_nn_match`、`make_features`、`resolve_hfnet_backend`、`load_ego_mask`、`apply_ego_mask_bgr`、`filter_kpts_by_mask`、`run_joint_features`；**预留** `HFNetONNX`（需 `onnxruntime`+权重，本版不启用） | `cv2`, `numpy`, `torch`, `lightglue`（经 `sfm_core`）；HF-Net 再加 `onnxruntime` |
| `sfm_localize.py` | `sfm_hfnet.py`（query 半部 + 缓存） | `localize_camera(...)`：`load_map_bundle` → 抽 query 帧 → `run_joint_features(sp_nv_nn)` → 逐帧 `localize_query` → 写 `errors.csv`/`query_meta.json`；`map_bundle_fingerprint`, `loc_ready`, `query_cache_valid` | `sfm_core`, `sfm_features` |
| `fusion.py` | `sfm_hfnet.py` / `query_demo_ekf.py` | `pose_confidence`, `align_by_nearest_stamp`, `fuse_se2_weighted`, `fuse_hard_select`, `fuse_loc_tables_se2`（两相机离线 se2，Option 2） | `numpy`, `pandas` |
| `motion_ekf.py` | `query_demo_ekf.py` | `wrap_pi`, `dead_reckon_se`, `interp_pose`, `interp_gt`, `propagate_pose`, `NoiseCfg`, `SE2EKF`, `build_sync_buckets`, `fuse_bucket_at_tlast`, `observation_R`, `s_of_pose`, `run_mode_a_se_predict`(Option1), `run_mode_b_ekf`(Option3), `write_process_noise_yaml`, `write_submission` | `numpy`, `pandas`, `yaml` |
| `obs_sigma.py` | `obs_error_sigma_s.py` | `load_params`, `sigma_of_s`（指向 `error-model/obs_error_sigma_vs_s.json`） | `numpy` |
| `metrics.py` | `sfm_hfnet.py` | `_rms`, `_wrap_deg`, `compute_camloc_metrics`（Option B 用 `tools/track_frame`），eval 专用 | `numpy`, `pandas` |
| `pipeline.py` | 新编排（参考 notebook cell 顺序） | `SubmissionConfig`(dataclass), `resolve_paths`, `ensure_assets`, `run_motion_only`, `run_two_cam`, `run_ekf`, `run_all` | 以上全部 |

> `track_frame.py` 不再单独重写——直接用 `tools/track_frame.py`（`assets_io`/`metrics` 通过把 `tools/` 加入 `sys.path` 导入 `load_lines`）。

### 5.1 地图：直接复用现成 `sp_nv_nn` bundle（本版不建图）

现有 `3d-map-test/outputs_hfnet_multi_lap_s/map_{fr,rl}_lap2+4/` 就是 `sp_nv_nn` 描述子的地图，与本版后端一致，**直接复制到 `sfm-model/`** 即可，无需重建。`sfm_localize.localize_camera` 加载它做 query。

> 不迁入建图代码（`project_frames_sn` / `build_map_s_neighbors` / `triangulate_pair` 等）。若未来换 HF-Net 或换建图圈需要重建，再单独补 `sfm_mapping.py`（见 §14）。

---

## 6. Lap 窗口来源（与 test 无 GT 的处理）

`lapdata/laps.json` 的 lap 窗口是**基于 GT 的起终点平面穿越**算出来的。含义：

- **eval 模式**：有 `gt.csv` ⇒ 可直接用 `laps.json`（或对 training mcap 重算）。`LAPS=None` 则遍历所有 lap；`LAPS=[3]` 只跑第 3 圈。每个 lap 用其 `[start_ns,end_ns]` 作为抽帧窗口，并裁出 `gt_lap{N}.csv` 供 evaluate。
- **predict 模式（test）**：**无 GT ⇒ 无法用平面穿越切圈**。设计策略（setup 可选）：
  1. `LAPS=None`（默认）：不按圈切，**整段 mcap** 跑一次，输出单个 `submission.csv`（coverage 以整段计）。
  2. 用户显式给时间窗：`TEST_WINDOWS=[(start_ns,end_ns), ...]` 手动指定每段的起止（例如赛方给定）。
  3. 若 test 侧也提供了一份 `laps.json`（比如用里程/其它信号切好），允许 `TEST_LAPS_JSON` 指向它。
- setup 里对 predict 模式默认取策略 1，并在 README 说明 2/3 的用法。

---

## 7. SETUP cell 规格（配置项）

```python
# ---- paths (原始 mcap；标定/GT/SE 从 mcap 内提取，无需额外目录) ----
TRAIN_MCAP = ".../c0_..._0.mcap"      # training 原始 mcap（含 /tf GT）
TEST_MCAP  = ".../test_xxx.mcap"      # test 原始 mcap（无 /tf GT）

# ---- run target ----
MODE   = "eval"           # "eval" (training+GT, 出 metrics) | "predict" (test, 仅 submission)
LAPS   = None             # None=全跑; 或 [3] / [5,7]
WINDOWS    = None         # 可选: [(start_ns,end_ns),...] 手动时间窗（predict 无 GT 切圈时用）
LAPS_JSON  = None         # 可选: 外部 lap 窗口 json（覆盖自动切圈）

# ---- cameras / backend ----
CAMS       = ("fr", "rl")           # 两相机主线
REF_CAMERA = "fr"                   # sync/对齐锚点
BACKEND    = "sp_nv_nn"             # 本版主线；"hfnet_onnx" 需权重+重建地图（未来）

# ---- localize params ----
QUERY_STRIDE = 1
TOP_K        = 5
MAP_LAPS     = (2, 4)               # 地图圈（bundle 已固定）
OVERRIDE_QUERY  = False             # True=忽略 query 缓存强制重算

# ---- resources (self-contained defaults, 指向 submission-demo 内) ----
SFM_MODEL_DIR = "sfm-model"         # map_fr_lap2+4 / map_rl_lap2+4 / polygons.json
TRACK_MCAP    = "track-model/map.mcap"
SIGMA_JSON    = "error-model/obs_error_sigma_vs_s.json"
NOISE_JSON    = "error-model/se_step_residual_noise.json"
TOOLS_DIR     = "tools"             # extract_assets / extract_gt / evaluate / track_frame
OUT_ROOT      = "outputs"
```

派生：`DATA_MCAP = TRAIN_MCAP if MODE=="eval" else TEST_MCAP`；`HAS_GT = (MODE=="eval")`；`ASSETS_DIR = outputs/<mode>_<stem>/assets`（首跑从 mcap 提取并缓存）。

---

## 8. 三个 Option cell 的行为

所有 option 都先确保**视觉观测就绪**（即每相机 `errors.csv`，在线算或读缓存）。Option 1 其实不需要视觉，但为了与 2/3 共用时间轴（在 `t_last` 采样），仍会用两相机的 stamp 构造 sync buckets。

### Option 1 — motion prediction only（`motion_only/`）
- 调 `dead_reckon_se`：以首帧 pose（eval 来自 GT，predict 来自 SE 首帧 `state_estimation`）初始化，积分 `vx,vy,ax,ay,wz`。
- 在两相机 sync 的 `t_last` 采样 → `submission.csv`。
- eval：算 `compute_camloc_metrics` + 图；predict：仅落盘。
- **依赖 `state_estimation.csv`**：predict 的 test `extracted/` 必须含它（运动输入）。文档标注此前置。

### Option 2 — two-camera localization（`two_cam/`）
- 确保 `loc_<lap>_fr/errors.csv` 与 `loc_<lap>_rl/errors.csv`（在线 localize 或缓存）。
- `fuse_loc_tables_se2(tables, ref_camera="fr")`：nearest-stamp ω 加权 SE(2)（门控 Δxy≤5m, Δyaw≤15°），无 EKF、无时间外推。
- 取 `ok` 行 → `submission.csv`。eval 出 metrics（用 fused 表里的 gt_*）。

### Option 3 — EKF fusion（`ekf/`）
- `run_mode_b_ekf`：SE2EKF 以首帧 pose 初始化；对每个 sync bucket：`predict_to(t_last)` → 各相机 PnP pose 外推到 `t_last` → `fuse_se2_weighted` → `observation_R(s)=diag(σ²(s),σ²(s),σ_yaw²)` → `update`。
- 过程噪声来自 `config/process_noise.yaml`（由 `se_step_residual_noise.json` 的 MADσ 生成）。
- 输出 `submission.csv` + `sim/ekf_sim.csv`（每帧 ok/权重/σ/P 对角）。eval 出 metrics。

### 汇总 cell（eval only）
- `compare/metrics_compare.csv` 并排三方案；`traj_vs_gt.png`、`err_vs_time.png`、`err_vs_s.png`。
- predict 模式：汇总 cell 只打印三份 `submission.csv` 的路径与行数/coverage 估计，不算误差。

---

## 9. 在线 localize + 缓存（`sfm_localize.localize_camera`）

```text
localize_camera(cam, *, data_mcap, extracted, lap_window, out_loc_dir, map_dir,
                feats, calib, gt_or_none, track_line, query_stride, top_k,
                map_laps, override_query, polygons_path, has_gt):
  1. map_frames, points = load_map_bundle(map_dir)         # 预建 bundle，不重建
  2. fp = map_bundle_fingerprint(map_dir)
     if (not override_query) and query_cache_valid(out_loc_dir, fp, lap, stride, top_k):
         return read_csv(out_loc_dir/errors.csv)            # 缓存命中，跳过
  3. frames = extract_query_frames(data_mcap, lap_window, calib, gt_or_none, cam, stride)
  4. run_joint_features(frames, feats, ego_mask=load_ego_mask(cam, polygons))
  5. rows=[]; for q in frames: r=localize_query(q, map_frames, points, feats, calib.K, calib.T_base_cam, top_k); rows.append(...)
  6. df=errors table; df.to_csv(errors.csv); write query_meta.json; save localize_times_s.npy
  7. return df
```

缓存键（写入 `query_meta.json`，复用 `query_cache_valid` 逻辑）：`map_fingerprint(mtime_ns+size)` + `data_mcap` 指纹（**新增**，因为换 mcap 结果必须失效）+ `query_lap` + `query_stride` + `top_k`。相对原实现**增加 mcap 指纹**，确保 train/test 不同 mcap 不会误命中彼此缓存。

errors.csv schema（与现有一致）：
```
stamp_ns, ok, gt_x, gt_y, gt_yaw, pred_x, pred_y, pred_yaw, err_xy, err_yaw_deg, n_pnp_corr, n_pnp_inl
```
predict 模式下 `gt_*` 与 `err_*` 为 `nan`（不影响 `ok`/`pred_*`）。

---

## 10. 自包含涉及的资源复制清单（生成代码时执行）

| 目标 | 来源 | 备注 |
|------|------|------|
| `sfm-model/map_fr_lap2+4/`, `map_rl_lap2+4/` | `3d-map-test/outputs_hfnet_multi_lap_s/map_{fr,rl}_lap2+4/`（全 4 文件） | sp_nv_nn 地图，直接复用 |
| `sfm-model/polygons.json` | `fr-auto-mask/polygons.json` | ego mask |
| `sfm-model/models/README_HFNET.md` | `3d-map-test/models/README_HFNET.md` | 未来 HF-Net 升级说明 |
| `track-model/map.mcap` | `c0_.../23-00-38_cut_120m_update/map.mcap` | racing line（track_frame 读取）|
| `error-model/obs_error_sigma_vs_s.json` | `3d-map-test/.../loc_lap3_fused/obs_error_sigma_vs_s.json` | sp_nv_nn 标定，匹配本版后端 |
| `error-model/se_step_residual_noise.json` | `state-error-estimation/se_step_residual_noise.json` | 过程噪声 MADσ（与后端无关）|

> `map.mcap` / `map_bundle.npz`（~400 MB/相机）体积大。建议加入 `.gitignore`，用软链接或运行时现读。生成代码前会就「复制 vs 软链接」与你确认。

---

## 11. 运行方式（README 草案）

```bash
# 依赖：torch, lightglue, numpy, pandas, opencv-python, scipy, pyyaml,
#       mcap, mcap-ros2, tqdm, (可选 onnxruntime)
# 1) 编辑 run_submission.ipynb 的 SETUP cell（选 MODE / 路径 / LAPS）
# 2) 顺序执行 SETUP → Option1 → Option2 → Option3 →（eval）汇总
# 3) 产物在 outputs/<mode>_<mcap>/{motion_only,two_cam,ekf}/submission.csv
```
首跑会在线 SfM localize（慢）；二跑相同任务直接读 `outputs/.../loc/.../errors.csv` 缓存（快）。

---

## 12. 需要确认的开放点

1. **产物形态**：主入口 **notebook**（`run_submission.ipynb`，setup + 3 option cell）为主，是否再加薄 CLI `run_submission.py`？
2. **大文件**：`map_bundle.npz`（~400 MB/相机）/ `map.mcap` 复制进 `submission-demo/` vs 软链接，倾向哪种？
3. **lap 切分（predict）**：无 GT 时默认整段跑（§6 策略 1）可接受吗？还是你会提供 `WINDOWS` / `LAPS_JSON`？

---

## 13. 实施顺序（确认后）

1. 资源落位：`map_{fr,rl}_lap2+4`、`map.mcap`、σ(s)、噪声、polygons 到 `sfm-model` / `track-model` / `error-model`。
2. 底层：`assets_io.py`（调 tools 提 mcap 资源）、`mcap_io.py`、`obs_sigma.py`。
3. SfM：`sfm_core.py` → `sfm_features.py`（sp_nv_nn 主线）→ `sfm_localize.py`（含缓存，复用现成地图）。
4. 定位方案：`fusion.py`、`motion_ekf.py`、`metrics.py`。
5. 编排：`pipeline.py` + `config/process_noise.yaml` 生成。
6. 入口：`run_submission.ipynb`（+ 可选 CLI）。
7. 自测：eval 模式 training lap3 跑通，与现有 notebook 对齐量级；predict 模式对一段 mcap 出 submission。

---

## 14. 未来升级：HF-Net（本版不实现，预留）

当前仓库**无 `hfnet.onnx`**，现有地图是 `sp_nv_nn` 建的。本版因此固定 `sp_nv_nn`。将来升级 HF-Net 需同时满足：

1. **权重**：按 `README_HFNET.md` 从 SavedModel 转出 `hfnet.onnx`，放 `sfm-model/models/`。
2. **地图**：HF-Net 描述子与 sp_nv_nn 不兼容，须用 HF-Net **重建** `map_{fr,rl}_lap2+4`（补一个 `sfm_mapping.py`：抽帧→HF-Net 特征→s-order 三角化→`save_map_bundle`）。
3. **σ(s) 重标定**：HF-Net 误差分布不同，`error-model/obs_error_sigma_vs_s.json` 需用 HF-Net 的 fused 误差重算。

接口层面 `sfm_features.make_features("hfnet_onnx")` / `HFNetONNX` 已预留，切换只需补上述三项资源。
```
