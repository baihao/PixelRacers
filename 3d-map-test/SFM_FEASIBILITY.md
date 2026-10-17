# camera_r / camera_fl：Lap2 建图 + Lap3 检索定位可行性分析

> 目录：`PixelRacers/3d-map-test`  
> 主路径数据源：`/constructor0/sensor/camera_r/compressed_image` + `camera_info`  
> 扩展路径数据源：`/constructor0/sensor/camera_fl/compressed_image` + `camera_info`（与 `r` **同一套建图思路**，独立地图）  
> 建图：`laps.json` **lap 2**　｜　定位评测：`laps.json` **lap 3**  
> 本文只做可行性分析；`camera_r` 已有 notebook 验证，`camera_fl` 为并行扩展方案。

---

## 1. 目标（你想建什么）

### 阶段 A — 建图（Lap 2）

对后视相机 `camera_r`：

1. SuperPoint 提取局部特征  
2. **描述子歧义度**过滤  
3. 与 **前/后各 2 帧** 做 **LightGlue** 匹配  
4. **GT 位姿** + **极线 / RANSAC** 筛匹配  
5. 三角化得到稀疏 **SfM 3D 点地图**  
6. **NetVLAD** 全局描述子建检索库  

### 阶段 B — 定位评测（Lap 3，建图完成之后）

对第三圈每一查询帧（或抽样子集）`camera_r`：

1. SuperPoint 特征检测  
2. 描述子歧义度过滤  
3. NetVLAD 全局描述 → 在 Lap2 库中检索 **top-5 候选帧**  
4. 查询帧与 5 个候选分别做 **LightGlue**  
5. 基于 SfM 地图做定位（2D–3D 对应 + **PnP / RANSAC**），与 Lap3 GT 对比误差  

已知：图像与时间戳、内参 `K,D`、外参 `tf_static`、GT（开发评测用）。

| 资产（Lap2 产出） | Lap3 用法 |
|---|---|
| NetVLAD `{g_i}` + 帧索引 | 检索 5 帧候选 |
| 稀疏 3D 点 + 帧上观测 | LightGlue 后建立 2D–3D，PnP |
| 候选帧位姿 \(T_{\mathrm{map}\leftarrow\mathrm{cam}}\) | 初值 / 失败时回退 |

### 扩展 — `camera_fl`（与 `r` 平行、地图独立）

前后视共视极少，**不共享 3D 点 / NetVLAD 库**。对 `camera_fl` 重复阶段 A/B 全流程（详见 **§10**）：

1. Lap2：`fl` 去畸变 → SuperPoint → 歧义过滤 → ±2 LightGlue → GT 极线三角化 → NetVLAD 库  
2. Lap3：`fl` 同预处理 → top-5 → LightGlue → SfM 2D–3D PnP → 与 GT 比误差  
3. **先不融合**：独立出 `loc_lap3_r` / `loc_lap3_fl`，按时间对齐看总体指标与失败互补（§10.4）；融合留到确认有互补收益之后（§10.5）

---

## 2. 问题重定义：这是不是经典 SfM？

| | 经典 SfM（COLMAP 等） | 本方案 |
|---|---|---|
| 位姿 | 未知，由匹配估计 | **已知（GT）** |
| 尺度 | 常有歧义 | **米制（GT）** |
| 主难点 | 位姿漂移、闭环 | 匹配质量、外参误差、夜间重复纹理 |
| 实质 | Structure-from-Motion | **已知位姿的多视图三角化 / Mapping** |

结论：在 GT 可用的前提下，**可行性高于**盲 SfM。  
更准确的名称是：**GT-posed sparse reconstruction**（带 SfM 式的特征—匹配—三角化流程）。

任务文档也允许开发阶段使用 GT；若最终要提交纯视觉定位，此地图像作 **先验地图**，与“评测时不用 GT”不冲突。

---

## 3. 数据与几何条件

### 3.1 时间与帧量

| | Lap 2（地图） | Lap 3（查询） |
|---|---|---|
| `start_ns` | `1791918444832466282` | `1791918612426651869` |
| `end_ns` | `1791918612426651869` | `1791918736054678423` |
| 时长 | ≈ **167.6 s** | ≈ **123.6 s** |
| 路径长 | ≈ 2993 m | ≈ 3024 m |
| 约帧数 @10 Hz | ≈ **1600–1700** | ≈ **1200–1300** |

Lap2 ±2 匹配边 \(O(4N)≈7\times10^3\)；Lap3 每查询 ×5 候选 LightGlue，全圈约 \(6\times10^3\) 次匹配，体量可控。  
两圈路径接近、同相机后视 → **外观可重访**，适合做跨圈定位测试（比同圈自检索更接近真实）。

### 3.2 相机 `r`（相对 fl/fr 的优势）

来自数据集 README Accuracy notes：

- **Pitch**：全相机可靠  
- **Yaw**：对 **`r` 和 `rl` 可靠**（fl/fr 约 ±1°）  
- **Roll / translation**：未标定（安装值）

`camera_r` 安装大致：车尾后方（`t_x≈-2.28 m`），光轴接近水平、朝后。  
相对前视双目：

- 朝向标定更可信 → **极线约束更可靠**  
- 后视常看到护栏、标线、地面结构，不完全依赖灯塔  
- 仍有 roll、相机高度/纵向安装误差 → 三角化有系统偏差，但通常可接受作稀疏地图

### 3.3 位姿链（每帧相机在 map 下）

\[
T_{\mathrm{map}\leftarrow\mathrm{cam}}
=
T_{\mathrm{map}\leftarrow\mathrm{base}}(t)
\;
T_{\mathrm{base}\leftarrow\mathrm{cam}}
\]

- \(T_{\mathrm{map}\leftarrow\mathrm{base}}\)：GT（`gt.csv` / SE），精度约 **0.1 m / 0.2°**  
- \(T_{\mathrm{base}\leftarrow\mathrm{cam}}\)：`tf_static`  

两帧 \(i,j\) 的相对位姿：

\[
T_{i\leftarrow j}
=
T_{\mathrm{cam}_i\leftarrow\mathrm{map}}
\;
T_{\mathrm{map}\leftarrow\mathrm{cam}_j}
\]

由此可写本质矩阵 / 投影矩阵，做极线与三角化，**无需** `recoverPose` 估位姿。

### 3.4 运动基线（尺度来源）

相邻帧 \(\Delta t≈0.1\,\mathrm{s}\)；第二圈车速若约 15–20 m/s，则：

| 匹配跨度 | 近似基线 |
|---|---|
| ±1 帧 | ≈ 1.5–2.0 m |
| ±2 帧 | ≈ 3–4 m |

远大于 fl–fr 安装基线（≈0.36 m），**深度几何条件更好**。  
低速/停车段基线变短 → 深度方差变大，需按 \(\|t\|\) 或视差门槛丢弃。

---

## 4. 拟定流水线（与需求对齐）

```text
lap2 camera_r 图像
    ↓
去畸变（K,D）
    ├────────────────────────────────────────────┐
    ↓                                            ↓
SuperPoint 局部特征                      NetVLAD 全局描述子 g_i
    ↓                                            ↓
描述子歧义度过滤                         归一化 + 存库（HDF5/NPZ）
    ↓                                            ↓
±2 帧 LightGlue 匹配 + GT 极线/RANSAC   帧索引 ↔ stamp ↔ T_map_cam
    ↓                                            （供之后 top-K 检索）
三角化 → map 系稀疏点云
```

局部建图与全局描述**共享同一套去畸变图像与时间戳**，一次遍历可并行提取，避免两套预处理不一致。

### 4.1 SuperPoint

- 输入：去畸变后的 `camera_r`（可 resize 加速，如长边 512–1024）  
- 输出：每帧 \(K_i, D_i\)（描述子）  
- 风险：夜间仍会在灯上扎堆 → 必须靠下一步过滤

### 4.2 描述子歧义度过滤（推荐具体做法）

对单帧描述子集合 \(\{d_k\}\)：

1. 对每个 \(k\)，在**同图内**找最近邻 \(k'\)（排除自身）  
2. 若 \(\|d_k-d_{k'}\| < \tau_{\mathrm{dup}}\)（过于相似）→ 丢弃或降权  
3. 或保留 uniqueness 分数 \(\|d_k-d_{k'}\|\)，只留 Top 分位  

可选叠加：高亮灯心掩膜、网格配额（每格最多 N 点）。  
过滤在 **LightGlue 之前**完成，减少歧义点进入匹配器。

### 4.3 时间匹配窗口 ±2 + LightGlue

- **匹配算法：LightGlue**（`features="superpoint"`），与 SuperPoint 官方配套；仓库可用 `lightglue` / hloc 接口  
- 输入：过滤后的 keypoints + descriptors；输出：匹配索引与置信度  
- 可按匹配分再截断（如 score > 0.1），再送入极线 / RANSAC  
- 窗口小 → 外观变化小、重叠大，匹配率应高于任意帧对  
- 注意：纯前后视运动时，远处点视差小，近处（护栏）更有用  
- 不采用裸 NN 作为主路径（仅可作消融对照）

### 4.4 GT + 极线 / RANSAC

已知 \(R,t\)（由 GT 推出）时：

\[
E = [t]_\times R,\quad
F = K^{-T} E K^{-1}
\]

筛法：

1. **极线距离** \(|\mathbf{x}'^\top F\mathbf{x}| / \|\cdot\| < \tau_{\mathrm{px}}\)（如 1–2 px）  
2. 可选再对“内点”做一次 RANSAC 估 \(E\) 做对照（诊断 GT/外参是否一致）  
3. 三角化后：**两侧深度 > 0**、重投影误差 < \(\tau_{\mathrm{reproj}}\)、深度 ∈ \([z_{\min}, z_{\max}]\)

这比“盲 RANSAC 估位姿再三角化”更稳，也更符合“GT 已知”的设定。

### 4.5 建 3D 模型

- 每条通过筛选的匹配 → 两点三角化得 \(X\)（map 或 cam0 系，最后统一到 map）  
- 多帧观测同一轨迹：用描述子跟踪或像素邻域关联 → 融合（均值 / 中位数）并记录观测次数  
- 输出：  
  - `points3D`（xyz + color/可选）  
  - `observations`（point_id, frame_id, uv）  
  - 可选导出 COLMAP / Open3D / PLY 便于可视化  

### 4.6 NetVLAD 全局描述（检索定位用）

**目的**：在线定位时，不先对全地图做局部匹配，而是用查询图 \(q\) 的全局向量在地图库中找 **top-K 最相似关键帧**，再用这些帧上的局部特征 / 关联 3D 点做精定位。

**建库（mapping，lap2 `camera_r`）**

1. 对每帧去畸变图 \(I_i\) 跑 NetVLAD（或 hloc 中的 `netvlad` / Pitts30k 权重）  
2. 得到 \(g_i \in \mathbb{R}^{D}\)（常见 \(D=4096\)，可 PCA 到 256/512）  
3. L2 归一化：\(g_i \leftarrow g_i / \|g_i\|\)  
4. 存储一行记录：

| 字段 | 含义 |
|---|---|
| `frame_id` / `stamp_ns` | 与局部特征、GT 对齐 |
| `g_i` | 全局描述子 |
| `T_map_cam` 或 `xyzyaw` | 检索命中后的位姿初值 |
| `image_path` | 可选，便于调试 |

实现上可直接用仓库已有 **hloc**（`third_party/Hierarchical-Localization`）的 `extract_features` + NetVLAD 配置，输出 HDF5，与 COLMAP/hloc 检索流程兼容。

**夜间赛道注意**

- 灯阵导致 **感知混叠**：必须 **top-5 + 几何验证**，不单信 top-1  
- 库与查询同为 `camera_r`、同一去畸变  
- 可选：Lap2 库按弧长抽关键帧，减轻相邻帧冗余  

---

## 5. 阶段 B：Lap3 定位流程（建图完成之后）

```text
Lap3 查询帧 q（camera_r）
    ↓
去畸变（与建图相同 K,D）
    ↓
┌──────────────────────┬──────────────────────┐
↓                      ↓
SuperPoint 检测        NetVLAD → g_q
↓                      ↓
描述子歧义度过滤         与 Lap2 库 {g_i} 余弦/点积
↓                      ↓
局部特征 (k_q, d_q)     top-5 候选帧 {c1…c5}
└──────────┬───────────┘
           ↓
对每个候选 c_k：
    LightGlue(q, c_k)
    → 匹配像素 → 经 c_k 的观测表升为 2D–3D（X in map）
    → 可选：极线/重投影预筛
           ↓
合并 5 个候选的 2D–3D 对应（去重 point_id）
           ↓
solvePnPRansac（K_r）→ T_cam←map 或 T_map←cam
           ↓
转到 base_link：T_map_base = T_map_cam · inv(T_base_cam)
           ↓
与 Lap3 GT 比 err_xy / err_yaw（仅评测）
```

### 5.1 查询端预处理（与建图一致）

1. **SuperPoint** 检测（resize 等超参与 Lap2 相同，避免域差）  
2. **描述子歧义度过滤**（同阈值策略）  
3. **NetVLAD** 得 \(g_q\) 并 L2 归一化  

### 5.2 检索：固定 top-5

- 相似度：\(s_i = g_q^\top g_i\)（已归一化时即余弦）  
- 取 **K = 5** 最高分 Lap2 关键帧  
- 用 GT 可离线报 **Recall@5**（命中定义：候选中至少一帧与 query GT 距离 < 5 m 或 10 m，且 yaw 差 < 30°）

### 5.3 LightGlue + 基于 SfM 的定位

对每个候选 \(c_k\)：

1. LightGlue 匹配 \((u_q, u_{c})\)  
2. 查 SfM 观测：\(u_c\) 若对应 map 点 \(X\)，则得到 query 上的 **2D–3D**：\((u_q, X)\)  
3. 若某候选关联 3D 点过少：可退化为 q–\(c_k\) 的 **2D–2D + 已知 \(T_{c_k}\)** 做相对位姿，再链式到 map（次选）  

合并 5 候选对应后：

- `cv2.solvePnPRansac`（或 SQPnP），得到相机位姿  
- 内点数、重投影误差作成功门槛（如 inl ≥ 12，med reproj < 3 px）  
- 失败则该 query 记为 lost / 用最佳候选帧位姿作极弱回退（评测时单独统计）

**注意**：定位阶段 **不使用 Lap3 GT 参与估计**，GT 只用于算误差；与任务“推理不用 GT”一致（首帧初始化若需要可另议）。

### 5.4 跨圈难点（Lap2 地图 → Lap3 查询）

| 问题 | 说明 | 缓解 |
|---|---|---|
| 轨迹不完全重合 | 横向偏移、速度不同 | top-5；放宽命中距离 |
| 光照/曝光微差 | 同夜录制通常仍接近 | 同预处理 |
| 地图点稀疏 | 部分候选无足够 3D | 多候选合并对应；降低歧义过滤强度做消融 |
| 感知混叠 | 5 候选全在错误直道 | PnP 内点门槛拒绝；统计 failure rate |

---

## 6. 可行性判断

### 6.1 总体结论

| 维度 | 判断 | 说明 |
|---|---|---|
| Lap2 建图几何 | **高** | GT 位姿 + 时间基线数米级 |
| 标定可用性 | **中高** | `r` 的 yaw/pitch 较有利；roll/t 为安装值 |
| Lap2 局部匹配 | **中** | 夜间灯阵；歧义过滤 + 极线 |
| Lap3 检索 R@5 | **中高** | 同相机跨圈重访；固定 K=5 + PnP 验证 |
| Lap3 SfM 定位 | **中（值得做）** | 依赖地图点密度与检索是否进对路段 |
| 工程体量 | **可控** | 建图 ~7k 边；定位 ~1.2k×5 次 LightGlue |

**综合：Lap2 建图 + Lap3 top-5 检索定位整条链路可行，建议作为主验证路径。**

### 6.2 主要风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 重复灯导致错配 | 建图飞点 / 定位错配 | 歧义过滤 + 高亮抑制 + 极线 |
| roll / 安装平移误差 | 点云与 PnP 偏差 | 监控重投影；必要时小外参精炼 |
| NetVLAD 混叠 | top-5 全错路段 | PnP 内点门槛；统计 failure |
| 跨圈轨迹偏移 | 共视变少 | 合并 5 候选 2D–3D；放宽评测距离 |
| 地图点过稀 | PnP 失败 | 放宽建图深度范围；增加观测次数阈值 |
| 低速建图基线短 | 深度差 | \(\|t\|<\tau_t\) 跳过 |

### 6.3 与“真 SfM（不估位姿）”对比

- 建图不做位姿优化：实现简单、尺度准。  
- 定位用的是 **地图点 + PnP**，不是再跑一遍 SfM。  
- 一期自研三角化 + PLY + 检索定位即可。

---

## 7. 成功标准（建议）

### 7.1 Lap2 建图

1. 极线后中位内点占比 > 30%（或 > 50 内点/边）  
2. 三角化中位重投影 < 1–2 px；深度合理（如 2–80 m）  
3. 点云沿赛道走廊分布，无明显飞点云  

### 7.2 Lap3 检索（K=5）

1. **Recall@5**：候选中至少一帧与 query GT 距离 < 5 m（或 10 m）且 yaw 差 < 30°；目标 R@5 尽量 > 0.7–0.8（以实测为准）  
2. R@1 可一并记录，但不作为唯一指标  

### 7.3 Lap3 SfM 定位

1. **成功率**：PnP 通过门槛的 query 比例（如 inl≥12）  
2. **误差**（成功样本）：中位 `err_xy`、中位 `|err_yaw|`；可与“只用 SE 速度积分”对照  
3. **失败分析**：检索失败 vs 匹配不足 vs PnP 炸掉，分类统计  

---

## 8. 建议实施顺序

**建图（Lap2）**

1. 抽 20–50 帧：SuperPoint → 歧义过滤可视化  
2. NetVLAD 烟测（邻帧高相似 / 远距低相似）  
3. \(i,i+1\) 极线 + 三角化  
4. ±2 窗口短轨迹 PLY  
5. 全圈 Lap2：`points3D` + `global_descriptors.h5` + `frames.csv` + 观测表  

**定位（Lap3）**

6. 提取 Lap3 图像（或按 stride 抽样）  
7. 每帧：SuperPoint → 歧义过滤 → NetVLAD → **top-5**  
8. 对 5 候选 LightGlue → 2D–3D → PnP  
9. 相对 Lap3 GT 出 `err_xy` / `err_yaw` 曲线与表格  
10. （可选）加 `rl`、多圈地图、关键帧子采样消融  
11. **扩展 `camera_fl`**：复用同一脚本/超参，仅换相机话题与 `T_base_fl` / `K_fl`（§10）  
12. （可选）Lap3 双相机位姿融合评测（§10.4）  

目录建议：

```text
3d-map-test/
  SFM_FEASIBILITY.md
  data/lap2_r/                 # 建图图像 (camera_r)
  data/lap3_r/                 # 查询图像 (camera_r)
  data/lap2_fl/                # 建图图像 (camera_fl)
  data/lap3_fl/                # 查询图像 (camera_fl)
  outputs/
    map_r/                     # points3D, observations, frames, netvlad (r)
    map_fl/                    # 同上，独立地图 (fl)
    loc_lap3_r/                # top5, poses, errors.csv
    loc_lap3_fl/
    loc_lap3_compare/          # 对齐对比：并集成功率、失败互补（先做）
    loc_lap3_fused/            # 可选：确认互补后再做
  scripts/
```

---

## 9. 结论

- **Lap2**：`camera_r` + GT 位姿稀疏建图 + NetVLAD 库 — **可行**（已有全圈实验）。  
- **Lap3**：同相机查询 → SuperPoint → 歧义过滤 → NetVLAD → **固定 5 候选** → LightGlue → **SfM 2D–3D PnP** — **可行，且是完整闭环验证**。  
- **扩展**：`camera_fl` 用**相同建图思路**在 Lap2 独立建图、Lap3 独立 query — **可行**；与 `r` **不共享特征/点云**。  
- **评测策略**：**先不融合**，对齐比较两路总体成功率 / 误差，并算「至少一路成功」看能否弥补单相机失败（§10.4）；有明显互补后再做位姿融合（§10.5）。  
- 定位阶段不把 Lap3 GT 喂进估计器，只用于评分。  
- 主风险仍是夜间混叠与地图点密度；`r` 实测失败主因是局部地图过稀与 PnP 拒识，前视有望覆盖不同纹理区。  
- 推荐交付物：`map_r` + `loc_lap3_r`；`map_fl` + `loc_lap3_fl`；**双路对齐对比表**（并集成功率）；融合报告可选、靠后。

---

## 10. 扩展：`camera_fl` Lap2 建图 + Lap3 查询

> 与 §1–§5 的 `camera_r` 流程**算法同构**，仅替换相机流、内参、外参与输出目录。  
> 前后视几乎无特征交集 → **必须独立 SfM / 独立 NetVLAD**，禁止把 `fl` 特征配进 `map_r`（或反之）。

### 10.1 为何加前视、为何独立建图

| 点 | 说明 |
|---|---|
| 互补视野 | `r` 朝后，`fl` 朝前偏左；失败区常与后视纹理/地图空洞相关，前视可提供另一组路标 |
| 无共视 | 前后特征交集≈空 → 联合三角化、共享描述子库无几何依据 |
| 位姿仍统一 | 两套地图的 3D 点都在同一 `map` 系（各自 \(T_{\mathrm{map}\leftarrow\mathrm{cam}}=T_{\mathrm{map}\leftarrow\mathrm{base}}\,T_{\mathrm{base}\leftarrow\mathrm{cam}}\)） |
| 标定差异 | README：`fl`/`fr` 的 **yaw 约 ±1°**（`r`/`rl` 更稳）；pitch 仍可靠。极线阈值宜略松于 `r`（见附录 B，可从 3–6 px 起调） |

`camera_fl` 安装大致：车头前方偏左（`t≈(0.94, 0.20, 0.34)` m），与 `r`（尾后 \(t_x≈-2.28\) m）空间分离。

### 10.2 阶段 A' — Lap2 建 `Map_fl`（同思路）

流水线与 §4 **逐模块相同**，输入改为 Lap2 的 `camera_fl`：

```text
lap2 camera_fl 图像
    ↓
去畸变（K_fl, D_fl）
    ├────────────────────────────┐
    ↓                            ↓
SuperPoint + 歧义过滤          NetVLAD → {g_i^fl}
    ↓                            ↓
±2 LightGlue + GT 极线         存库 netvlad_db_fl
    ↓
三角化 → points3D_fl（map 系）
```

实现要点：

1. **时间窗 / 圈次**：与 `r` 相同，`laps.json` lap 2 的 `[start_ns, end_ns]`。  
2. **位姿链**：\(T_{\mathrm{base}\leftarrow\mathrm{fl}}\) 取 `tf_static` 的 `base_link→camera_fl`；GT 仍是 `base_link` 的 `xyzyaw`。  
3. **超参起点**：与 `r` 烟测/全圈已验证值对齐（如 `SP_RESIZE=512`、`SP_MAX_KP`、`MIN_NN_DIST`、`MATCH_WINDOW=±2`、`EPI_THRESH_PX`）；若极线内点率明显偏低，优先略增 `EPI_THRESH_PX`（yaw 噪声）。  
4. **输出目录**：`outputs/map_fl/`（`points3D.ply`、`observations`、`netvlad_db.npz`、`frames`），与 `map_r` **物理隔离**。  
5. **成功标准**：沿用 §7.1（内点率、重投影、走廊状点云）；额外记录相对 `r` 的点密度沿弧长分布，便于看互补性。

### 10.3 阶段 B' — Lap3 用 `fl` 查询 `Map_fl`

与 §5 **同构**，查询流仅为 Lap3 `camera_fl`：

```text
Lap3 q_fl
  → 去畸变 / SuperPoint / 歧义过滤 / NetVLAD
  → 仅在 Map_fl 的 {g_i^fl} 上 top-5
  → LightGlue(q_fl, c_k^fl) → 2D–3D（仅 points3D_fl）
  → solvePnPRansac(K_fl) → T_map_cam_fl
  → T_map_base = T_map_cam_fl · inv(T_base_fl)
  → 与 Lap3 GT 比 err_xy / err_yaw
```

约束：

- 检索库、候选帧、3D 观测表 **全部来自 `Map_fl`**。  
- 定位不用 Lap3 GT；GT 仅评分。  
- 评测表与 `r` **同一字段**（`stamp_ns`、`ok`、`n_pnp_corr`、`n_pnp_inl`、`err_xy`、`err_yaw`），便于 §10.4 对齐。  
- 成功标准沿用 §7.2–§7.3；本阶段**不做位姿融合**。

### 10.4 先不融合：双路独立结果对比与互补性（当前优先）

目标：看 `fl` 总体是否可用，以及能否**补上 `r` 过多的失败**，再决定要不要做融合。

**时间对齐**（`r`/`fl` 的 `header.stamp` 差约 µs，可忽略运动）：

1. 以 Lap3 时间轴为基准，对每个 `r` 查询帧取最近邻 `fl` 帧（或反过来；\(|\Delta t|\) 中位应 ≪ 1 ms）。  
2. 得到逐时刻二元组 \((\mathrm{ok}_r,\mathrm{ok}_{fl},\mathrm{err}_r,\mathrm{err}_{fl})\)。  
3. 输出 `outputs/loc_lap3_compare/pair_errors.csv`。

**总体指标（各报一行）**

| 指标 | `camera_r` | `camera_fl` |
|---|---|---|
| 成功率 \(\mathrm{ok}\) | 已测约 **57%**（709/1235） | TBD |
| 中位 `err_xy` / p95（成功子集） | 已有 `errors.csv` | TBD |
| 中位 \|err_yaw\| | 同上 | TBD |
| Recall@5（可选） | 同上 | TBD |

**互补性（回答「能不能弥补单路失败」）**

设 \(A=\mathrm{ok}_r\)，\(B=\mathrm{ok}_{fl}\)（对齐后逐帧布尔）：

| 量 | 定义 | 解读 |
|---|---|---|
| \(P(A)\) | `r` 成功率 | 后视基线 |
| \(P(B)\) | `fl` 成功率 | 前视是否单独可用 |
| \(P(A\cup B)\) | **至少一路成功** | 若显著高于 \(\max(P(A),P(B))\) → 有互补 |
| \(P(B\mid \neg A)\) | **`r` 失败时 `fl` 仍成功** | 直接衡量「救失败」能力 |
| \(P(A\mid \neg B)\) | `fl` 失败时 `r` 仍成功 | 对称 |
| \(P(\neg A\land \neg B)\) | 双失败率 | 融合/时间滤波也难救的硬区 |
| 仅 `r` / 仅 `fl` / 双成功 | 计数与占比 | 画在轨迹上：失败区是否错开 |

判定经验（可按任务改）：

- 若 \(P(B\mid\neg A) \gtrsim 0.3\) 且 \(P(A\cup B)-P(A) \gtrsim 0.1\) → **值得后续做选优/融合**；  
- 若 \(P(A\cup B)\approx P(A)\)（`fl` 失败与 `r` 高度重叠）→ 前视对补失败帮助有限，优先加厚地图而非融合；  
- 若 \(P(B)\) 本身明显低于 `r`（标定/纹理更差）但仍有较高 \(P(B\mid\neg A)\) → 仍可作「失败时的备胎」，融合用硬选优即可。

**可视化（建议）**

1. 轨迹图：`r` 失败点、`fl` 失败点、双失败点分色；  
2. 沿圈时间线：两路 `ok` 条带 + 并集条带；  
3. 对 `r` 的集中失败区（已有 `fail_regions`），逐区报 \(P(B\mid\neg A)\)。

**`camera_r` 现状（对照基线，Lap3 全圈）**

- 硬失败约 **43%**（526/1235），主因地图局部过稀 + PnP 拒识；NetVLAD 几乎不是主因。  
- 集中失败区多段连续数十～百帧 → 若 `fl` 在这些弧段上 \(P(B\mid\neg A)\) 高，则双相机价值明确。

### 10.5 多相机位姿融合（`r` / `fr` / `rl`）

独立 SfM 各自得到 `base_link` 位姿后，在**位姿层**融合（不融特征、不拼 2D–3D）：

\[
\omega = \mathbf{1}[\mathrm{ok}] \cdot n_{\mathrm{inl}} \cdot
\exp\!\big(-\bar{e}_{\mathrm{reproj}} / \sigma\big)
\quad(\sigma\sim 4\,\mathrm{px})
\]

若暂无重投影误差，可退化为 \(\omega=\mathbf{1}[\mathrm{ok}]\cdot n_{\mathrm{inl}}\)。

| 模式 | 做法 |
|---|---|
| 硬选优 | 取 \(\omega\) 最大的一路 |
| SE(2) 加权 | \(\alpha_c=\omega_c/\sum\omega\)；\(x,y\) 线性、yaw 圆周平均 |
| 门控 | 与最高权位姿差 \(>5\,\mathrm{m}\) 或 \(>15^\circ\) 则丢弃该路，防错融 |

时间戳：`r`/`fr`/`rl` 差约 µs，最近邻对齐即可。  
实现与评测：`3d-map-test/sfm_hfnet.py`（`fuse_multi_camera_tables`）+ notebook `sfm_hfnet_lap2_map_lap3_loc.ipynb` §4。

### 10.6 可行性与风险（相对 `r`）

| 维度 | 判断 | 说明 |
|---|---|---|
| 算法复用 | **高** | 与 `r` 同代码路径，换相机即可 |
| 几何 / GT | **高** | 同样 GT-posed 三角化；时间基线数米级 |
| 标定 | **中** | yaw ±1° → 极线/三角化略逊于 `r` |
| 外观 | **中高** | 前视纹理不同，**是否补失败由 §10.4 实测** |
| 工程量 | **约 ×2** | 建图+定位再跑一路；对比脚本很轻 |

**综合：先完成 `Map_fl` + 独立 Lap3 query，用 §10.4 并集/条件成功率回答互补性；融合不是当前目标。**

### 10.7 建议实施顺序（fl，无融合）

1. 复用 `sfm_experiment`：参数化 `camera_name ∈ {r, fl}`（话题、`K,D`、`T_base_cam`、输出前缀）。  
2. Lap2 短片段烟测 `fl`（特征 → ±2 三角化 → PLY）。  
3. Lap2 全圈 `map_fl` + NetVLAD。  
4. Lap3 全圈定位 → `loc_lap3_fl/errors.csv`（字段对齐 `loc_lap3_r`）。  
5. **对齐对比（§10.4）**：算 \(P(A),P(B),P(A\cup B),P(B\mid\neg A)\) + 轨迹/失败区图 → 结论是否值得融合。  
6. （仅当 5 为正面）再实现 §10.5。

---

## 附录 A：关键数据路径

| 资源 | 路径 |
|---|---|
| mcap | `c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap` |
| 话题 `r` | `/constructor0/sensor/camera_r/compressed_image` |
| 话题 `fl` | `/constructor0/sensor/camera_fl/compressed_image` |
| 内参 | `extracted/camera_info.json` → `r` / `fl` |
| 外参 | `extracted/tf_static.json` → `base_link→camera_r` / `base_link→camera_fl` |
| GT | `extracted/gt.csv`（`base_link` in `map`） |
| 建图圈 | `laps.json` → **lap 2** |
| 定位圈 | `laps.json` → **lap 3** |

## 附录 B：极线阈值直觉

焦距 \(f≈1491\,\mathrm{px}\)。若相对姿态误差约 \(0.2°\)（GT 量级）：

\[
\delta \approx f \cdot \tan(0.2^\circ) \approx 5\,\mathrm{px}
\]

外参再叠一点误差时，极线阈值宜从 **2–3 px** 起调，过严会杀真匹配，过松会放进灯阵错配。应用内点率与重投影联合选 \(\tau\)。

对 **`camera_fl`**：yaw 标定约 ±1° 时，同量级估计 \(\delta \approx f\cdot\tan(1^\circ)\approx 26\,\mathrm{px}\) 为上界直觉；实践中 GT+安装误差通常小于该上界，建议阈值从 **3–6 px** 起（与当前 `r` 实验的 6 px 同量级），再按内点率微调。
