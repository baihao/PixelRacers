# submission-demo

自包含的两相机（fr + rl）定位提交代码。输入原始 `.mcap`，支持 **eval**（training + GT，出 metrics）与 **predict**（test，仅出 submission），提供三种定位方案。

设计说明见 [`DESIGN.md`](DESIGN.md)。

## 后端

本版用 `sp_nv_nn`（SuperPoint 局部特征 + NetVLAD 全局 + mutual-NN 匹配），与 `sfm-model/map_{fr,rl}_lap2+4/` 预建地图一致，直接复用、不重建。HF-Net 为未来升级（见 `DESIGN.md` §14）。

## 依赖

```
torch, lightglue, numpy, pandas, opencv-python, scipy, pyyaml,
mcap, mcap-ros2-support, tqdm
```
Kernel：`pixel-racers`（`/Users/hbai/SchubKI/PixelRacers/pixel-racers`）。NetVLAD 走 `third_party/Hierarchical-Localization`（首用自动下载 Pitts30K 权重）。

> macOS 上若遇 `OMP: Error #15`，代码已在入口设 `KMP_DUPLICATE_LIB_OK=TRUE`；手动跑脚本前也可 `export KMP_DUPLICATE_LIB_OK=TRUE`。

## 运行

### Notebook（主入口）
打开 `run_submission.ipynb`，编辑 **SETUP** cell（`MODE` / mcap 路径 / `LAPS`），顺序执行：
SETUP → Option 1（motion only）→ Option 2（two-camera）→ Option 3（EKF）→ 汇总。

### CLI

```bash
cd submission-demo
export KMP_DUPLICATE_LIB_OK=TRUE

# eval：training mcap，全部圈，三个方案
python run_submission.py --mode eval --train-mcap /path/c0_..._0.mcap

# eval：只跑 lap3，只跑 EKF
python run_submission.py --mode eval --train-mcap ... --laps 3 --options ekf

# 快速冒烟（抽帧 stride 20）
python run_submission.py --mode eval --train-mcap ... --laps 3 --query-stride 20

# predict：test mcap（无 GT，默认整段 mcap，仅出 submission）
python run_submission.py --mode predict --test-mcap /path/test.mcap
```

## 三种方案（option）

| option | 目录 | 说明 |
|---|---|---|
| `motion_only` | `outputs/<mode>_<stem>/<win>/motion_only/` | SE 航位积分，在两相机 t_last 采样 |
| `two_cam` | `.../two_cam/` | fr+rl 在线定位 → nearest-stamp ω 加权 SE(2) 融合（无 EKF）|
| `ekf` | `.../ekf/` | SE 预测 + fr/rl 外推 → se2 → EKF update，R(s)=σ(s)² |

每个方案输出 `submission.csv`（`timestamp_ns,x,y,yaw`）；eval 另出 `metrics_official.csv`。

## 产物布局

```
outputs/<mode>_<mcap_stem>/
  assets/                 # 从 mcap 提取：camera_info/tf_static/track_bounds/state_estimation(+gt eval)
  loc/<win>/loc_{fr,rl}/  # 在线定位结果 errors.csv + query_meta.json（缓存）
  <win>/motion_only|two_cam|ekf/submission.csv [+ metrics_official.csv]
  compare/metrics_compare.csv    # eval 汇总
```

## 缓存

`loc/<win>/loc_<cam>/errors.csv` 带 `query_meta.json` 指纹（地图 + mcap + 窗口 + stride + top_k + backend）。相同任务二次运行直接复用，不重跑特征/PnP。强制重算：SETUP 里 `OVERRIDE_QUERY=True` 或 CLI `--override-query`。

## 资源目录（自包含）

| 目录 | 内容 |
|---|---|
| `sfm-model/map_{fr,rl}_lap2+4/` | sp_nv_nn 预建地图 bundle |
| `sfm-model/polygons.json` | fr/rl ego mask |
| `track-model/map.mcap` | racing line（(s,n) 帧）|
| `error-model/obs_error_sigma_vs_s.json` | 观测噪声 σ(s) |
| `error-model/se_step_residual_noise.json` | EKF 过程噪声 MADσ |
| `config/process_noise.yaml` | 由上一项首次运行自动生成 |
| `tools/` | `extract_assets.py` / `extract_gt.py` / `convert.py` / `evaluate.py` / `track_frame.py` |

## predict（test）注意

- 无 GT ⇒ 不算 metrics、不切圈：默认整段 mcap 出一份 submission；需要分段可用 `WINDOWS` / `--windows start:end`，或 `LAPS_JSON`。
- 运动输入 `state_estimation`（vx/vy/ax/ay/wz）在 test 里也有（它是车端 SE，不是 GT），所以 Option 1/3 可用，首帧初值取 SE 首帧 pose。
