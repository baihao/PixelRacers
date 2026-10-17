# PixelRacers

Constructor Labs 赛道相机定位实验：mcap 回放、HF-Net SfM 建图/定位、SE 过程噪声、Lap3 EKF query-demo。

## 1. 环境

```bash
cd PixelRacers
python3.12 -m venv pixel-racers
source pixel-racers/bin/activate          # 或: source scripts/env.sh
python -m pip install -U pip
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m ipykernel install --user --name pixel-racers --display-name "pixel-racers"
```

Notebook 一律选 kernel **`pixel-racers`**。Apple Silicon 上 SfM 强制 **MPS**。

可选：若要用真正 HF-Net ONNX，见 [`3d-map-test/models/README_HFNET.md`](3d-map-test/models/README_HFNET.md)；默认后端 `sp_nv_nn`（SuperPoint + NetVLAD + mutual NN）即可跑通。

## 2. 数据依赖（仓库外）

默认 bag / 导出资产应放在与本仓库同级：

```text
SchubKI/
  c0_2026-10-13_22-53-42_camloc/
    c0_2026-10-13_22-53-42_camloc_0.mcap
    extracted/gt.csv
    extracted/state_estimation.csv
    extracted/track_bounds.json
    23-00-38_cut_120m_update/map.mcap   # Option B racing line
  PixelRacers/                          # 本仓库
```

圈次时间窗：`lapdata/laps.json`。

产物（map / loc / 图）默认写在各子目录的 `outputs*` 下，已被 `.gitignore` 忽略，需本机跑 notebook 生成。

## 3. 组件一览

| 路径 | 做什么 |
|------|--------|
| `camloc_play.py` | 读 mcap 相机 JPEG（无需完整 ROS 2） |
| `lapdata/` | 各圈 `start_ns` / `end_ns` |
| `odometry/` | SE 的 v/a/wz 航位推算基线 |
| `state-error-estimation/` | SE 逐步残差 → 过程噪声 MADσ |
| `3d-map-test/` | Lap2+4 建图 + Lap3 多相机定位 + σ(s) |
| `query-demo/` | Lap3：纯 SE vs EKF（三相机 sync + se2） |
| `fr-auto-mask/polygons.json` | `camera_fr` 车体 mask |
| `third_party/Hierarchical-Localization` | hloc submodule |

## 4. 怎么跑（推荐顺序）

### 4.1 快速试播 mcap

```bash
source pixel-racers/bin/activate
python camloc_play.py --list-topics
python camloc_play.py --camera fl --first
# GUI（需桌面）:
# python camloc_play.py --camera fl --play --duration 5
```

Notebook：`test/play_mcap.ipynb`。

### 4.2 航位基线（可选）

打开 `odometry/lap2_velocity_odometry.ipynb`，自上而下跑完。  
用 SE 的 `vx,vy,ax,ay,wz` 从圈起点积分，在相机时间戳上与 GT 对比。

### 4.3 过程噪声 MADσ（EKF 需要）

打开 `state-error-estimation/se_step_residual_noise.ipynb`，跑完生成：

- `se_step_residual_noise.json`
- `se_step_residual_noise_summary.csv`

query-demo 会读其中的 **`mad_sigma`**（不是样本 std）。仓库里已带一份可用的 JSON；改数据后请重跑本 notebook。

### 4.4 SfM 建图 + Lap3 定位（视觉观测）

主 notebook：

```text
3d-map-test/sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb
```

设计说明：`3d-map-test/SFM_MULTI_LAP_S_ORDER.md`。

| 步骤 | 内容 |
|------|------|
| §1–2 | 加载标定 / GT / laps / HF-Net（或 `sp_nv_nn`） |
| § 建图+query | 相机 `r` / `fr` / `rl`：Lap**2+4** 按 $s$ 建图 → Lap**3** query |
| § metrics | 各相机 `metrics_official` + 轨迹 |
| § 融合 | 三相机 `fuse_se2` → `outputs_hfnet_multi_lap_s/loc_lap3_fused/` |
| §6–7 | 多高斯 $\sigma(s)$ → `obs_error_sigma_vs_s.json`（EKF 观测噪声） |

输出目录（体积大，已 ignore）：

```text
3d-map-test/outputs_hfnet_multi_lap_s/
  map_{r,fr,rl}_lap2+4/
  loc_lap3_{r,fr,rl}/
  loc_lap3_fused/obs_error_sigma_vs_s.json
```

首次全量较慢。缓存有效时会跳过 map/query；强制重跑设 notebook 里 `OVERRIDE_MAP` / `OVERRIDE_QUERY = True`。

烟测可把 `MAP_STRIDE` / `QUERY_STRIDE` 改为 `2`。

### 4.5 Lap3 EKF query-demo

```text
query-demo/lap3_ekf_query_demo.ipynb
```

设计：`query-demo/QUERY_DEMO_EKF.md`。

依赖（先跑 4.3 / 4.4，或使用已有产物）：

- `state-error-estimation/se_step_residual_noise.json`
- `3d-map-test/outputs_hfnet_multi_lap_s/loc_lap3_{r,fr,rl}/errors.csv`
- `3d-map-test/outputs_hfnet_multi_lap_s/loc_lap3_fused/obs_error_sigma_vs_s.json`
- `extracted/state_estimation.csv`、`gt.csv`、`track_bounds.json`、`map.mcap`

Notebook 流程：

1. 写 `query-demo/config/process_noise.yaml`（MADσ）
2. **Mode A** `se_predict`：仅 SE 航位
3. **Mode B** `ekf_cam`：SE 预测 + 三相机齐套 → 外推到 $t_{\mathrm{last}}$ → se2 → EKF（$R=\sigma(s)$）
4. **对比** `se_predict` / `fused_se2` / `ekf_cam` → metrics + 轨迹图

产物：

```text
query-demo/outputs/lap3/
  gt_lap3.csv
  se_predict/submission.csv
  ekf_cam/submission.csv
  compare/metrics_official.csv
  compare/metrics_official.png
```

官方评测（必须用裁剪后的 Lap3 GT）：

```bash
python ../c0_2026-10-13_22-53-42_camloc/tools/evaluate.py \
  query-demo/outputs/lap3/gt_lap3.csv \
  query-demo/outputs/lap3/ekf_cam/submission.csv
```

## 5. 常见问题

| 现象 | 处理 |
|------|------|
| `No module named lightglue` / 缺依赖 | `source pixel-racers/bin/activate` 后重装 `requirements.txt`；确认 submodule 已 init |
| OpenMP 重复库报错 | `export KMP_DUPLICATE_LIB_OK=TRUE`（`scripts/env.sh` 已设） |
| evaluate coverage &lt; 90% | 勿用全行程 `gt.csv`；用 `query-demo/outputs/lap3/gt_lap3.csv` |
| query-demo 找不到 loc / σ(s) | 先跑通 `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb` §建图到 §7 |
| MPS 不可用 | SfM notebook 要求 Apple GPU；无 MPS 需改代码放宽设备（当前硬要求） |

## 6. Submodule

```bash
git submodule update --init --recursive
# 上游: https://github.com/cvg/Hierarchical-Localization.git
```
