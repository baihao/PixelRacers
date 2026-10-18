# PixelRacers 摄像头定位 — 5 页 PPT 内容说明

> 用途：先用本 MD 定稿每页文案、公式、图表与数据来源，再据此做 PPT。  
> 数据来源：`paper/1812.03506v2.pdf`、`3d-map-test/`、`odometry/`、`state-error-estimation/`、`query-demo/`。  
> 评测圈默认：**Lap3**（建图用 Lap2 / Lap2+4）；EKF 官方分：**Lap5–7**（fr+rl）。

---

## 全局版式建议

| 项 | 建议 |
|---|---|
| 页数 | **5 页**（原理 / 相机实验 / 运动与误差 / EKF / 改进） |
| 风格 | 深色赛道氛围或浅色工程风均可；每页 **1 主标题 + 1 副句**，避免卡片堆叠 |
| 字体 | 标题用有辨识度的无衬线；公式用 LaTeX 渲染后贴图 |
| 图源 | 优先用仓库已有 png（路径见各页「建议插图」） |

---

# 第 1 页｜基于摄像头 stream 的定位算法详解

## 标题
**分层视觉定位：从全局检索到 6-DoF 位姿**

副标题：HF-Net 层级定位（Sarlin et al., arXiv:1812.03506）与本项目单相机 SfM 地图实现

## 本页要讲清楚的故事（口述提纲）

1. **问题**：仅用车载多路相机 JPEG stream（~10 Hz），估计 `base_link` 在 `map` 下的 \((x,y,\psi)\)。  
2. **经典解法（论文）**：粗到细（Coarse-to-Fine）层级定位，而非直接对全库做稠密匹配。  
3. **本车约束**：标定与安装下 **不宜双目**；改为 **每相机独立 SfM 地图 + 在线 query**。

## 正文块 A — 论文原理（1812.03506）

论文标题：**From Coarse to Fine: Robust Hierarchical Localization at Large Scale**（HF-Net）。

**流水线（板书用）：**

```text
Query 图像
  → (1) 全局描述子（NetVLAD / 全局特征）→ 在地图关键帧库中 Top-K 检索
  → (2) 局部特征检测与描述（SuperPoint / HF-Net local）
  → (3) 与候选关键帧做 mutual NN 匹配
  → (4) 2D–3D 对应 + PnP（+ RANSAC）→ 相机位姿
  → (5) 外参变换到 base_link → (x, y, yaw)
```

**为何分层：**  
大规模场景下，直接全库局部匹配不可行；全局检索把搜索空间压到少数关键帧，再在局部做精确几何求解，兼得鲁棒性与效率。

**与本项目的对应实现（`3d-map-test/sfm_hfnet*.py` / notebook）：**

| 论文步骤 | PixelRacers 实现 |
|---|---|
| 全局检索 | NetVLAD top-K（邻域可用 racing-line \(s\) 排序 / arc 邻域） |
| 局部特征 | HF-Net ONNX 或 `sp_nv_nn`（MPS） |
| 匹配 | mutual NN + 比率检验 |
| 几何 | OpenCV PnP + inlier 统计 |
| 地图 | 离线对选定圈次做 SfM / 三角化，缓存 `map_{cam}_lap*/map_bundle.npz` |

## 正文块 B — 本项目为何不用双目、而用「每相机独立 SfM」

**标定与安装事实（写入 PPT 要点）：**

- 七路相机中，**相对更可信的是 `fl, fr, r, rl`**（其余侧向相机标定偏差更大）。  
- 即便这几路：**内参、朝向、焦距差异大**（如 `fl`/`fr` 窄视场焦距差异显著），**立体基线与外参不足以支撑稳定双目深度 / 立体 VO**。  
- `fl` 与 `fr` 画面高度重叠，但 `fl` 视场更窄 → 可检出特征更少。

**因此采用的架构：**

```text
离线（每相机独立）:
  图像序列_c + GT pose  →  SfM / 3D landmarks  →  map_c

在线（每相机独立 query）:
  当前帧_c  →  对 map_c 做层级定位  →  pose_c
  （多相机再做 SE(2) 融合 / EKF，见第 2、4 页）
```

要点一句话：**不把多相机当立体对，而把每路相机当成「带自己地图的单目定位传感器」。**

## 建议插图

| 位置 | 内容 |
|---|---|
| 左上 | 论文式 Coarse→Fine 流程图（自制） |
| 右上 | 车端相机布局示意（标出 fl/fr/r/rl） |
| 底部 | 「独立 map_c」三栏示意：fr / r / rl 各一张 query 与轨迹片段 |

## 页脚来源
`paper/1812.03506v2.pdf` · `PixelRacers/3d-map-test/`

---

# 第 2 页｜定位情况分析与摄像头选择

## 标题
**选哪些相机？单圈建图 vs 双圈建图？如何融合？**

副标题：Lap3 query · 地图 Lap2 vs Lap2+4 · 单相机 vs se2 融合

## 正文块 A — 相机筛选逻辑

1. 候选集：`fl, fr, r, rl`（标定相对可靠）。  
2. **`fl` 剔除**：与 `fr` 场景高度相似，但视场窄、特征点更少 → 性价比低。  
3. **实验三相机：`fr, r, rl`**（与 `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb` 一致）。

## 正文块 B — 建图策略对比（Lap2-only vs Lap2+4）

评测：**同一 Lap3 query**。  
对照 run：`lap2_time`（仅第二圈建图） vs `lap2+4_s`（第二圈+第四圈、按 \(s\) 序建图）。

**官方风格 metrics（成功帧上的位置 RMSE；括号内 success rate）：**

| Camera | Lap2-only A-RMSE [m] | success | Lap2+4 + s-order A-RMSE [m] | success |
|---|---:|---:|---:|---:|
| **r** | 7.10 | 42.3% | **2.89** | **69.4%** |
| **fr** | 7.65 | 72.9% | **4.16** | **90.4%** |
| **rl** | 18.59 | 56.7% | **7.42** | **72.2%** |

**结论（写在图注下）：** 多圈地图 + racing-line \(s\) 邻域显著降低 RMSE、提高成功率；后续一律用 **`map_*_lap2+4`**。

## 正文块 C — 三相机融合算法（简述）

对齐到同一 query 时间轴后：

1. 每路得到 \(\hat{z}_c=(x_c,y_c,\psi_c)\) 与置信权重  
   \(\omega_c \propto \mathbf{1}[\mathrm{ok}_c]\cdot n_{\mathrm{inl},c}\cdot e^{-\bar{e}_{\mathrm{reproj}}/\sigma}\)。  
2. **`fuse_se2_weighted`**：在 SE(2) 上按 \(\omega\) 加权融合；门控 \(\Delta xy\le 5\,\mathrm{m}\)、\(\Delta\psi\le 15^\circ\)。  
3. 对照：**hard** = 只取 \(\omega\) 最大一路。

## 正文块 D — 单相机 vs 融合（Lap3，map=lap2+4）

| 方法 | success | pos RMSE [m] | within 1 m（相对成功帧） | within 1 m（相对全部 query） |
|---|---:|---:|---:|---:|
| r | 69.4% | 2.89 | 58.3% | 40.5% |
| fr | 90.4% | 4.16 | 72.2% | 65.3% |
| rl | 72.2% | 7.42 | 81.0% | 58.5% |
| **fused_se2** | **96.4%** | **4.42** | **75.4%** | **72.7%** |
| fused_hard | 96.4% | 4.45 | 73.7% | 71.1% |

**解读：**  
- 融合把 **coverage/成功率** 拉到 ~96%，整体 within-1m（相对全部帧）明显好于任一路。  
- RMSE 介于最好单路与最差单路之间：融合优先「更常有解 + 抑制离群」，而非单纯压低已成功帧的 RMSE。

## 建议插图（本页必须有轨迹）

| 图 | 路径 |
|---|---|
| 单相机轨迹 | `3d-map-test/outputs_hfnet_multi_lap_s/loc_lap3_{r,fr,rl}/traj_vs_gt.png` |
| 融合轨迹 | `.../loc_lap3_fused/traj_fused_se2.png` |
| Lap2 vs Lap2+4 柱状图 | notebook 导出的 compare 图 / `metrics_official_compare.csv` 重画 |
| 单路 vs 融合柱状图 | `.../loc_lap3_fused/metrics_fuse_compare.png` |
| 误差–里程 | `.../loc_lap3_fused/err_vs_s_fused_se2.png` |

## 页脚来源
`sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb` · `outputs_hfnet_multi_lap_s/`

---

# 第 3 页｜运动预测误差 × 图像定位误差（\(s\)）

## 标题
**两套不确定度：动力学航位 vs 视觉定位**

副标题：为何要「用实验」标定视觉噪声 —— 特征检测/匹配误差难以解析建模

## 正文块 A — 动力学位置预测（参考 `odometry/`）

**输入：** `state_estimation` 的 \(v_x,v_y,a_x,a_y,\omega_z\)（**不用** SE 的 \(x,y\) 当观测；仅首帧 pose 可初始化）。

**积分（body → map）：**

\[
\Delta \mathbf{p}_{\mathrm{body}} = \mathbf{v}\,\Delta t + \tfrac12 \mathbf{a}\,\Delta t^2,\quad
\psi \leftarrow \psi + \omega_z\Delta t
\]

位移用区间中点航向旋到 `map`；每步用测量 \(\mathbf{v}\) **重置速度**，避免 \(a\) 双重积分发散。

**实验量级（Lap2、以 FL stamp 采样的航位示例，`lap2_velocity_odometry.ipynb`）：**

| 模式 | xy median [m] | xy p95 [m] | 终点误差 [m] |
|---|---:|---:|---:|
| \(v\) only | ≈ 8.37 | ≈ 22.9 | ≈ 15.7 |
| \(v+a\) | ≈ 8.26 | ≈ 23.0 | ≈ 15.7 |

**结论：** 纯航位在一圈尺度上会漂到 **数米～十余米**，必须靠视觉修正；航位适合做 **高频预测**，不适合单独当最终定位。

## 正文块 B — Motion 逐步噪声（参考 `state-error-estimation/`）

对 SE 上报量与 pose 有限差分做残差，用稳健统计：

\[
\mathrm{MAD}\sigma = 1.4826 \cdot \mathrm{median}\big(|r-\mathrm{median}(r)|\big)
\]

写入 EKF 过程噪声的锚点（全行程）：

| 通道 | MADσ | 单位 |
|---|---:|---|
| \(\sigma_{v_x}\) | 0.073 | m/s |
| \(\sigma_{v_y}\) | 0.020 | m/s |
| \(\sigma_{a_x}\) | 10.59 | m/s² |
| \(\sigma_{a_y}\) | 8.63 | m/s² |
| \(\sigma_{\omega_z}\) | 0.0031 | rad/s |

建议插图：`state-error-estimation/se_step_residuals_vs_gaussian.png`、`se_step_residuals_qq.png`。

## 正文块 C — 图像定位误差随跑道里程 \(s\)（参考 multi-lap notebook）

在 racing line 弧长 \(s\) 上统计融合/单路误差，拟合 **\(\sigma_{\mathrm{obs}}(s)\)**（多高斯 / bin-MAD，再 sharpen），供 EKF 观测协方差使用。

建议插图：

- `loc_lap3_fused/obs_error_sigma_vs_s.png`  
- `loc_lap3_fused/sigma_multigauss_vs_s.png`  
- `loc_lap3_fused/err_vs_s_fused_se2.png`

## 正文块 D — 为什么必须用实验确定图像定位误差？

写在醒目位置（引用用户原意，稍作工程化）：

> 视觉误差 **不止** 来自安装/内参偏差与成像随机噪声，更主要来自：  
> **特征检测误差、描述子噪声、跨帧匹配错误、PnP 外点** —— 这些环节高度非线性、场景相关，**很难用解析公式先验量化**。  
> 因此用 **Lap3 query × GT** 在 \(s\) 上 empirically 估计 \(\sigma(s)\)，再进入滤波器。

## 正文块 E — 为降低 detect / matching 误差已做的工作（条列）

- HF-Net / SuperPoint 类局部特征 + NetVLAD 全局检索（层级缩小误匹配空间）  
- mutual NN + 比率检验；PnP RANSAC + inlier 计数进融合权重  
- **`fr` ego-mask**（自车区域多边形），减少引擎盖/车身误特征  
- 多圈地图（Lap2+4）与 **按 \(s\) 的邻域选关键帧**，减少外观歧义  
- 融合门控（位置/航向一致性）抑制单路炸点  

## 页脚来源
`odometry/lap2_velocity_odometry.ipynb` · `state-error-estimation/se_step_residual_noise.ipynb` · `loc_lap3_fused/obs_error_sigma_*`

---

# 第 4 页｜EKF：运动预测 + 图像校正

## 标题
**扩展卡尔曼滤波：高频航位预测 × 异步视觉更新**

副标题：噪声来自第 3 页；流程与公式见 `QUERY_DEMO_EKF.md`；结果见 Lap5–7 fr+rl demo

## 正文块 A — 系统角色

| 输入 | 角色 |
|---|---|
| SE：\(v,a,\omega_z\) @ ~106 Hz | **Predict** |
| 相机 fr / r / rl @ ~10 Hz（实验最终用 **fr+rl**） | **Localize → Fuse → Update** |
| 首帧 SE pose | **仅初始化** |
| GT / SE pose 序列 | **禁止作运行时观测** |

> **相机取舍说明（与第 2 页一致）：** 最终 EKF 采用 **`fr` + `rl`**（非 `fl`）。  
> 相对三相机：少一路可省算力；相对 `r+rl`：`fr+rl` 在 Lap3/5 上精度–时延更均衡（见 query-demo 两相机对照）。

## 正文块 B — 完整数学流程（建议半页公式区）

**状态：**

\[
\mathbf{x}=[x,\;y,\;\psi]^\top
\]

**1. 预测（运动模型 \(f\)，控制 \(u=(v_x,v_y,a_x,a_y,\omega_z)\)）**

\[
\mathbf{x}_{k|k-1}=f(\mathbf{x}_{k-1|k-1},u_k,\Delta t),\quad
P_{k|k-1}=F_k P_{k-1|k-1} F_k^\top + Q_k
\]

\[
Q \approx G\,\mathrm{diag}(\sigma_{v_x}^2,\sigma_{v_y}^2,\sigma_{a_x}^2,\sigma_{a_y}^2,\sigma_{\omega}^2)\,G^\top
\]

（\(\sigma_\bullet\) = 第 3 页 MADσ）

**2. 异步相机 → 对齐到 \(t_{\mathrm{last}}\)**

\[
t_{\mathrm{last}}=\max_c t_c,\quad
\hat{z}_c(t_{\mathrm{last}})=\mathrm{Propagate}\big(\hat{z}_c(t_c),\,u,\,t_c\to t_{\mathrm{last}}\big)
\]

**3. 融合观测**

\[
z=\mathrm{fuse\_se2}\big(\{\hat{z}_c(t_{\mathrm{last}})\},\{\omega_c\}\big)
\]

**4. 更新（观测噪声随里程）**

\[
\hat{s}=\mathrm{xy\_to\_s}(\mathbf{x}_{k|k-1}),\quad
R(\hat{s})=\mathrm{diag}\big(\sigma_{\mathrm{obs}}^2(\hat{s}),\,\sigma_{\mathrm{obs}}^2(\hat{s}),\,\sigma_\psi^2\big)
\]

\[
K=P H^\top(HPH^\top+R)^{-1},\quad
\mathbf{x}\leftarrow\mathbf{x}+K(z-h(\mathbf{x})),\quad
P\leftarrow(I-KH)P
\]

\(h(\mathbf{x})=\mathbf{x}\)；yaw 创新 wrap 到 \((-\pi,\pi]\)。视觉失败则 **只预测、不更新**（保 coverage）。

**流程简图：**

```text
SE(v,a,ω) ──predict──► x,P
                         │
相机帧 ──localize──► z_c ──propagate──► fuse_se2 ──update(R(s))──► x,P
```

## 正文块 C — 实验结果（官方 `evaluate.py`）

方法：**ekf_cam_2cam（fr+rl）** · 地图 Lap2+4 · 评测 Lap5 / 6 / 7  
submission：`query-demo/outputs/lap5_7_2cam_fr_rl/submissions/`

| 窗口 | poses | coverage | **pos RMSE [m]** | lat / lon RMSE | yaw RMSE [deg] | within 1 m |
|---|---:|---:|---:|---|---:|---:|
| Lap5 | 1523 | 100% | **1.116** | 0.849 / 0.724 | 0.445 | 73.3% |
| Lap6 | 1205 | 100% | **0.875** | 0.260 / 0.835 | 0.215 | 81.9% |
| Lap7 | 1095 | 100% | **0.867** | 0.385 / 0.777 | 0.204 | 77.4% |
| Lap5–7 合并 | 3823 | 100% | **0.976** | 0.592 / 0.776 | 0.325 | 77.2% |

**对照（同设定下，示意量级，Lap5）：**  
纯 SE 航位 RMSE ~11 m；离线 fr+rl 融合 ~1.8 m；**EKF ~1.1 m** —— 在精度与可在线运行之间取得平衡。

## 建议插图

| 图 | 路径 |
|---|---|
| 轨迹 | `query-demo/outputs/lap5_7_2cam_fr_rl/lap{5,6,7}/compare/traj_vs_gt.png` |
| 误差–时间 | `.../compare/err_vs_time.png` |
| 模式柱状图 | `.../compare/metrics_official.png` |
| 跨圈汇总 | `.../metrics_cross_lap.png` |
| 公式旁小图 | EKF predict/update 框图（自制） |

## 页脚来源
`query-demo/QUERY_DEMO_EKF.md` · `lap5_7_ekf_2cam_fr_rl_query_demo.ipynb` · `submissions/evaluate_metrics.txt`

---

# 第 5 页｜可改进方向

## 标题
**下一步：更稳的特征、更好的地图、更强的横向约束、更广的外观覆盖**

## 改进点（四条，做成 2×2 或纵向四段）

### 1. 他车 / 动态目标 Mask
在特征提取前做 **车辆检测**，将他车区域 mask 掉，避免把动态物体上的特征点配进地图，降低定位抖动与炸点。

### 2. 地图：建图圈 + 多圈优化（而非只堆关键帧）
当前用 **Lap2+Lap4 建 SfM**。可改为：  
- 仍用这两圈控制地图规模；  
- 另用 **Lap1 / Lap3 / Lap5** 等对地图做 **bundle adjustment / 路标优化**。  
→ **地图不大，但精度吃满多圈训练数据。**

### 3. 车道线 + 高精度 2D 赛道图
引入车道 / 路沿检测，结合已有 `track_bounds` 高精度二维图，加强对 **lateral offset（\(n\)）** 的约束，进一步压横向误差。

### 4. 外观域扩展（白天）
现有训练/建图几乎全是 **夜场**。需补充 **白天** 数据建图或域自适应，提升跨时段稳定性。

## 收尾一句
当前系统已在多圈正式指标上达到 **~0.9–1.1 m RMSE、coverage 100%**；上述四点分别对应 **鲁棒性、地图精度、横向精度、泛化** 四条主轴。

## 建议插图
- 左：失败案例截图（他车入画 / 夜场过曝）  
- 右：改进路线图（Mask → Map BA → Lane → Day）

---

## 附录：PPT 制作 checklist

- [ ] 第 1 页：论文流程图 +「每相机独立 SfM」架构图  
- [ ] 第 2 页：**必须有** `traj_vs_gt` / `traj_fused_se2`，并贴 Lap2 vs Lap2+4 与 fused 表  
- [ ] 第 3 页：航位误差曲线 + MADσ 直方图 + \(\sigma(s)\) 曲线  
- [ ] 第 4 页：EKF 公式区 + Lap5–7 轨迹 + official evaluate 表（强调 **fr+rl**，勿写成 fl）  
- [ ] 第 5 页：四条改进，一条一图标即可  

### 关键路径速查

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
