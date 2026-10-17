# query-demo：Lap3 EKF 定位仿真（设计稿）

> 状态：**仅设计文档，供 review；尚未实现代码。**  
> 目标：在仿真中按 **ROS2/mcap 到达顺序** 驱动定位，对比 **纯 SE 航位** 与 **EKF（航位 + r/fr/rl 视觉）**，输出 metrics / 轨迹图 / camloc submission，并通过 `evaluate.py`。

---

## 1. 目标与范围

| 项 | 选择 |
|---|---|
| 数据 | `c0_2026-10-13_22-53-42_camloc_0.mcap` |
| **评测 / Query 圈** | **Lap 3**（与 `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb` 一致） |
| **地图** | Lap2 + Lap4 建图：`outputs_hfnet_multi_lap_s/map_{r,fr,rl}_lap2+4`（**不重建**，直接加载） |
| 运动输入 | `state_estimation` 的 **vx, vy, ax, ay, wz**（**不用** SE 的 x,y 做观测；仅 **首帧 pose** 作初值） |
| 视觉输入 | `camera_r` / `camera_fr` / `camera_rl`（齐套后融合；**不做单相机中间步骤**） |
| 过程噪声 | `state-error-estimation/se_step_residual_noise` 的 **MADσ**（见 §5.2） |
| 观测噪声 | \(\sigma_{\mathrm{obs}}(s)\)：`obs_error_sigma_vs_s.json` / `obs_error_sigma_s.sigma_of_s` |
| 三相机融合 | `fuse_se2_weighted`（与 notebook 一致；可选 hard 对照） |
| 对比实验 | **A：纯 SE 预测** vs **B：EKF 预测 + 三相机观测修正** |
| 输出 | 两套 submission + metrics + 轨迹/误差图 + 仿真中间结果落盘 |

仿真语义：**按 mcap 消息时间线回放**，在「收到某 topic」时触发对应处理，逼近车端异步到达。

**为何是 Lap3 而非 Lap2：** SfM 地图由 Lap2+Lap4 建成，notebook 的 query/评测圈是 Lap3；query-demo 直接复用同一套地图与 query 设定，避免用建图圈自评。

---

## 2. 动机：为何要 EKF + 异步相机校正

离线 notebook 是「整圈 query → 融合 → 评测」。真车上：

1. **SE（~106 Hz）** 持续提供 v/a/ω，用于预测；  
2. **三路相机（~10 Hz）曝光时刻不同**，同一「逻辑帧」的 r / fr / rl 时间戳不一致；  
3. 必须等三路齐套后做视觉 innovation，且较早相机的位姿要 **外推到最后一路的 stamp** 再融合；  
4. 预测协方差随时间增长；观测协方差随赛道位置 \(s\) 变化（失败区更大）。

本 demo 把上述时序与不确定性写进统一 EKF，并与纯航位基线对比。

---

## 3. 输入 Topic 与角色

| Topic | 角色 | 备注 |
|---|---|---|
| `/constructor0/state_estimation` | **预测驱动** | 用 `vx,vy,ax,ay,wz`；`x,y,yaw` 仅用于 **第一帧初始化** 与评测 GT |
| `/constructor0/sensor/camera_{r,fr,rl}/compressed_image` | **观测** | `header.stamp` = 曝光时刻；提交 CSV 的 `timestamp_ns` 取该逻辑帧最后一路 stamp |
| `/tf_static`、`camera_info` | 标定 | 启动时加载一次 |
| `/tf` | **禁止作输入** | 仅评测 / 开发对照 |

Lap3 时间窗：与 `lapdata/laps.json` / multi-lap notebook 一致（`start_ns`…`end_ns`）。

---

## 4. 系统架构（事件驱动）

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

### 4.1 两种运行模式（必须都跑，并对比）

| Mode | 名称 | 行为 |
|---|---|---|
| **A** | `se_predict` | 仅 SE 航位积分；相机事件 **不触发 update**（可仍按 \(t_{\mathrm{last}}\) 采样输出，保证与 B 同时间轴） |
| **B** | `ekf_cam` | SE 预测 + 三相机齐套 → 外推 → se2 融合 → EKF update（\(R(s)\)） |

同一套 Lap3 时间窗、同一初始 pose、同一采样时间戳，便于直接对比。

### 4.2 同步键（同一帧）

相机约 10 Hz、三路 stamp 差通常在数 ms～数十 ms。首版建议：

- **时间窗聚类**：以最早到达的一路为锚，`|t - t_anchor| < τ_sync`（如 40–80 ms）归入同一 `SyncBucket`；  
- 或按 **最近邻配对**：以 `r` 为锚，为 fr/rl 各取最近 stamp（与现 `align_by_nearest_stamp` 类似）。

齐套条件：bucket 内 `r, fr, rl` 均有图；`t_last = max(t_r, t_fr, t_rl)`。

**不做单相机 EKF 中间实验**；观测路径直接是三相机 sync + se2。

---

## 5. 运动模型（参考 `odometry/`）

与 `lap2_velocity_odometry.ipynb` 的积分公式一致（应用到 Lap3 时间窗），body 系速度/加速度 + 偏航角速度：

\[
\Delta p_{\mathrm{body}} = v\,\Delta t + \tfrac12 a\,\Delta t^2,\quad
\psi \leftarrow \psi + \omega_z\Delta t
\]

位移用区间中点航向转到 `map`。实现上对每个 SE 间隔做一步积分；**每个 SE 用测量 \(v\) 重置速度状态**（避免 \(a\) 双重积分发散），加速度主要进入 \(\frac12 a\Delta t^2\) 与过程噪声。

### 5.1 EKF 状态（Mode B）

\[
\mathbf{x} = [x,\; y,\; \psi]^\top \in SE(2)
\]

首版 \(v,a,\omega\) 作 **控制输入 \(u\)**。Mode A 只积分 \(\mathbf{x}\)，可不维护 \(P\)；Mode B 维护：

\[
\mathbf{x}_{k|k-1} = f(\mathbf{x}_{k-1|k-1},\, u_k,\, \Delta t),\quad
P_{k|k-1} = F_k P_{k-1|k-1} F_k^\top + Q_k(\Delta t, u_k)
\]

### 5.2 过程噪声 \(Q\)（直接用 SE 逐步残差 MADσ）

**不做**「纯 odometry vs GT」标定。过程噪声取自：

- Notebook：`PixelRacers/state-error-estimation/se_step_residual_noise.ipynb`
- 导出：`se_step_residual_noise.json` / `se_step_residual_noise_summary.csv`

\[
\mathrm{MAD}\sigma = 1.4826 \cdot \mathrm{median}\big(|r - \mathrm{median}(r)|\big)
\]

残差含义：同一 SE 消息内「上报字段 − 由 pose 有限差分得到的参考量」。

**写入 `config/process_noise.yaml` 的锚点数值**（全行程、`LAP_FILTER=None`）：

| 通道 | 来源残差 | MADσ | 单位 | EKF 用途 |
|---|---|---|---|---|
| \(\sigma_{vx}\) | `r_vx`（SE − pose FD） | **0.0729** | m/s | 纵向速度过程噪声 |
| \(\sigma_{vy}\) | `r_vy` | **0.0204** | m/s | 横向速度过程噪声 |
| \(\sigma_{ax}\) | `r_ax`（\|v\|>1） | **10.59** | m/s² | 加速度二次项噪声 |
| \(\sigma_{ay}\) | `r_ay` | **8.63** | m/s² | 同上 |
| \(\sigma_{\omega}\) | `r_wz`（`wz` − d(yaw)/dt） | **0.00313** | rad/s | 航向角速度（**用 wz，不用 yaw_vel_rad**） |

说明：

- 航位积分用 **`wz_radps`**；角速度噪声用 `wz_minus_dyaw.mad_sigma`。  
- **一律用 MADσ**，不用样本 std。  
- 实现时 **读 JSON 的 `mad_sigma`** 生成 yaml，上表仅作文档锚点。

\[
Q \approx G\,\mathrm{diag}(\sigma_{vx}^2,\sigma_{vy}^2,\sigma_{ax}^2,\sigma_{ay}^2,\sigma_{\omega}^2)\,G^\top
\]

**禁止用 SE 的 pose 作为过程更新**；pose 真值只用于初始化第 0 帧与 scoring。

---

## 6. 视觉观测链（直接复用 multi-lap Lap3 query）

**地图与模型：直接加载 notebook 产物，不在 query-demo 内重建。**

| 资源 | 路径（相对 `3d-map-test/`） |
|---|---|
| 地图 | `outputs_hfnet_multi_lap_s/map_{r,fr,rl}_lap2+4/`（`map_bundle`） |
| Query 缓存（可选加速） | notebook 写入的 `query_meta` / 特征缓存（若存在则复用） |
| 观测 \(\sigma(s)\) | `outputs_hfnet_multi_lap_s/loc_lap3_fused/obs_error_sigma_vs_s.json` |
| 算法与超参 | `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb` / `sfm_hfnet.run_camera_multi_lap_s_pipeline` 的 localize 部分 |

对齐套的每一路相机 \(c\in\{r,fr,rl\}\)：

1. 去畸变 +（`fr`）ego mask；  
2. HF-Net / `sp_nv_nn` 特征（MPS）；  
3. 对 `map_{c}_lap2+4`：NetVLAD top-K → mutual NN → PnP；  
4. 得到 \(\hat{z}_c=(x_c,y_c,\psi_c)\) @ stamp \(t_c\)，以及 \(n_{\mathrm{inl}}\)、（可选）重投影误差。

### 6.1 时间校正到 \(t_{\mathrm{last}}\)（关键）

三路 stamp 不同。融合前，对每一路成功观测：

\[
\hat{z}_c(t_{\mathrm{last}})
= \mathrm{Propagate}\big(\hat{z}_c(t_c),\; u(\cdot),\; t_c\to t_{\mathrm{last}}\big)
\]

\(\mathrm{Propagate}\) **复用同一航位模型**（用 \([t_c,t_{\mathrm{last}}]\) 内 SE 的 v/a/ω）。  
PnP 失败的相机不进入融合。

### 6.2 三相机位姿融合 → 单次观测 \(z\)

在 \(t_{\mathrm{last}}\) 上：

\[
\omega_c = \mathbf{1}[\mathrm{ok}_c]\cdot n_{\mathrm{inl},c}\cdot\exp(-\bar{e}_{\mathrm{reproj}}/\sigma)
\]

- **se2**（默认）：`fuse_se2_weighted`（门控 \(\Delta xy\le5\,\mathrm{m}\), \(\Delta\psi\le15^\circ\)）  
- **hard**（可选对照）：取 \(\omega\) 最大一路  

得到 \(z=(x,y,\psi)_{t_{\mathrm{last}}}\)。**无有这一条视觉路径，无单相机 EKF 分支。**

---

## 7. EKF 更新与 \(R(s)\)（仅 Mode B）

预测位姿 \(\mathbf{x}_{k|k-1}\) 投影到 racing line 得 \(\hat{s}\)（`map.mcap` + `track_frame`）。

\[
R(\hat{s}) =
\mathrm{diag}\big(
\sigma_{\mathrm{obs}}^2(\hat{s}),\;
\sigma_{\mathrm{obs}}^2(\hat{s}),\;
\sigma_\psi^2
\big)
\]

\(\sigma_{\mathrm{obs}}(s)\) 来自 multi-Gaussian 导出；\(\sigma_\psi\) 用 notebook yaw 经验值或常数（如 \(1^\circ\sim2^\circ\)）。

\[
K = P H^\top (HPH^\top+R)^{-1},\quad
\mathbf{x}\leftarrow\mathbf{x}+K(z-h(\mathbf{x})),\quad
P\leftarrow (I-KH)P
\]

\(h(\mathbf{x})=\mathbf{x}\)；yaw 创新 wrap 到 \((-\pi,\pi]\)。

视觉融合失败：本周期 **只预测、不更新**，默认仍输出预测位姿以保 coverage≥90%。

---

## 8. 仿真时间线（伪代码）

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
          append submission (t_last, x, y, yaw)   # Mode A: x 仅为航位
          clear buffer
```

因果性：时刻 \(t\) 的输出只使用 stamp ≤ \(t\) 的图像与 SE。

---

## 9. 输出、对比与 Validation

### 9.1 Submission CSV（camloc README）

```
timestamp_ns,x,y,yaw
```

- `timestamp_ns`：该逻辑帧 **最后一路相机** 的 `header.stamp`  
- 频率：~10 Hz；coverage（1 s bins）**≥ 90%**  
- 路径：

```text
query-demo/outputs/lap3/
  gt_lap3.csv                 # 裁剪 GT（evaluate 必须用这个）
  se_predict/submission.csv
  ekf_cam/submission.csv
```

实现入口：`lap3_ekf_query_demo.ipynb` + `query_demo_ekf.py`。  
视觉观测默认复用 `loc_lap3_{r,fr,rl}/errors.csv`（同一 SfM query，避免 notebook 内重跑 HF-Net）。

### 9.2 评测

```bash
python3 c0_.../tools/evaluate.py outputs/lap3/gt_lap3.csv se_predict/submission.csv
python3 c0_.../tools/evaluate.py outputs/lap3/gt_lap3.csv ekf_cam/submission.csv
```

### 9.3 必须保存的对比产物

| 产物 | 说明 |
|---|---|
| `se_predict/submission.csv` | Mode A |
| `ekf_cam/submission.csv` | Mode B |
| `compare/metrics_compare.csv` | 两模式 official A/B metrics 并排（RMSE、within 1 m、coverage 等） |
| `compare/traj_vs_gt.png` | GT + Mode A + Mode B 同图 |
| `compare/err_vs_time.png` | 两模式 xy/yaw 误差 vs 时间 |
| `compare/err_vs_s.png` | 两模式误差 vs Option B \(s\) |
| `se_predict/` / `ekf_cam/` 各自 | `metrics_official.csv`、`traj_vs_gt.png`、`err_vs_time.png`、`err_vs_s.png` |
| `ekf_cam/sim/`（可选） | 每帧：`t_last`、各相机 ok/ω、融合前后 pose、创新、\(P\) 对角、\(\sigma(s)\) |

复用 `compute_camloc_metrics`（与 notebook 一致）生成 metrics；轨迹/误差图风格对齐 `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb`。

---

## 10. 目录（已实现）

```text
PixelRacers/query-demo/
  QUERY_DEMO_EKF.md
  README.md
  lap3_ekf_query_demo.ipynb  # 主入口
  query_demo_ekf.py          # SE/EKF/sync/metrics 库
  config/process_noise.yaml
  outputs/lap3/
    gt_lap3.csv
    se_predict/
    ekf_cam/ (+ sim/)
    compare/
```

复用：

- `PixelRacers/camloc_play.py` — mcap 迭代  
- `PixelRacers/odometry/lap2_velocity_odometry.ipynb` — 航位公式（用于 Lap3）  
- `PixelRacers/state-error-estimation/se_step_residual_noise.json` — 过程噪声 MADσ  
- `PixelRacers/3d-map-test/outputs_hfnet_multi_lap_s/map_*_lap2+4` — **现成地图**  
- `PixelRacers/3d-map-test/sfm_hfnet.py` — localize + `fuse_se2_weighted`  
- `PixelRacers/3d-map-test/obs_error_sigma_s.py` — \(R(s)\)  
- `23-00-38_cut_120m_update/tools/track_frame.py` — \(s,n\)

---

## 11. 实施顺序

1. ~~过程噪声配置~~ → `config/process_noise.yaml`  
2. ~~Mode A 纯 SE~~ → `outputs/lap3/se_predict/`  
3. ~~Mode B EKF+三相机~~ → `outputs/lap3/ekf_cam/`（观测用已有 `loc_lap3_*`）  
4. ~~对比落盘~~ → `outputs/lap3/compare/`  
5. （可选）在线 HF-Net query / 全 run

**明确不做：** 单相机观测 EKF 中间步骤。

---

## 12. 风险与开放点

| 点 | 说明 | 建议 |
|---|---|---|
| SE 是否算「允许输入」 | README 禁止用 pose；**v/a/ω 来自同一 SE 消息** | 首版按「类 IMU/轮速」使用；文档标明边界 |
| 同步窗 \(\tau_{\mathrm{sync}}\) | 过大混帧，过小丢齐套 | 从 50 ms 扫 |
| 外推误差 | \(t_{\mathrm{last}}-t_c\) 大时视觉外推也吃 \(Q\) | 可把外推不确定性加进该路观测协方差再融合 |
| 地图 / query 一致性 | 地图 Lap2+4，query Lap3 | 与 notebook 完全一致，无需改图 |
| fr mask / MPS | 与 notebook 一致 | 硬性沿用 |
| 实时性 | 全量 HF-Net 可能慢于 10 Hz | demo 可离线按事件顺序；优先复用 query 缓存 |

---

## 13. Review 清单

- [x] 评测 / Query 圈为 **Lap 3**（地图 Lap2+4，直接加载现成模型）  
- [x] **不做**单相机 EKF；观测路径 = 三相机 sync + 外推 + se2  
- [x] 先 **Mode A 纯 SE**，再 **Mode B EKF+相机**，对比 metrics + 轨迹并落盘  
- [ ] 运动输入：SE 的 **v/a/ω**；pose **仅首帧**  
- [ ] \(R=\sigma_{\mathrm{obs}}^2(s)\) 来自 multi-Gaussian 导出  
- [ ] Submission 格式与 `evaluate.py` / coverage≥90%  
- [ ] 过程噪声：`se_step_residual_noise` 的 **MADσ**（角速度用 `wz`）

确认后按 §11 实现 `query-demo` 代码与 Lap3 两模式跑通。
