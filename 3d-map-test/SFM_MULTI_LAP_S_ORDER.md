# camera_r：Lap2+Lap4 按赛道弧长 \(s\) 联合建图 + Lap3 定位（方案稿）

> 状态：**已实现 notebook** — `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb`  
> 辅助 API：`sfm_hfnet.py`（`require_mps` / `project_frames_sn` / `build_map_s_neighbors` / …）  
> 对照：`sfm_hfnet_lap2_map_lap3_loc.ipynb`（单圈时间邻域 ±2）。  
> \(s,n\)：`23-00-38_cut_120m_update/tools`（`track_frame.py`）。

---

## 1. 目标

| 项 | 选择 |
|---|---|
| 相机 | **`r` / `fr` / `rl`**（各自独立 map+query；`fr` 用 ego mask） |
| 特征 / 定位栈 | **HF-Net**（`sfm_hfnet.py`：优先 `hfnet_onnx`，否则代理 `sp_nv_nn`；匹配均为 mutual NN，**不用 LightGlue**） |
| 加速 | **强制启用 MPS**（Apple Silicon；`sfm.pick_device()` → `mps`；不可用时显式报错，不静默回退 CPU） |
| 建图圈 | **Lap 2 + Lap 4**（两圈关键帧并入同一 SfM / NetVLAD 库） |
| 查询圈 | **Lap 3**（不参与建图，避免训测泄漏） |
| 邻域定义 | 不再用「时间轴 ±k」，改用 **赛道弧长 \(s\) 排序后的索引邻域** |
| 匹配窗 | 对排序后的帧 \(i\)：匹配 \(i\pm1,\;i\pm2,\;i\pm3\)（最多 6 条边 / 帧） |
| 评测 | 与现 notebook 相同：**Option A/B metrics + 误差曲线 + 轨迹图** |

动机：单圈按时间 ±2 匹配时，跨圈、同位置的共视用不上；两圈并库后若仍按时间邻域，Lap2 末尾与 Lap4 开头在时间上不相邻。把帧投到 racing line 的 \(s\) 上再排序，**同弯道、同直道上的跨圈帧会变成邻居**，有望加厚原先稀疏失败区的 3D 点。

---

## 2. 为何必须 \(x,y \rightarrow s,n\) 再按 \(s\) 排序

两圈（或任意多圈）联合建图时：

1. 用 `map.mcap` 中的 **racing line**（`/constructor0/map_ar/racing_line`）把每帧 GT 的 \((x,y[,yaw])\) 投影为 \((s,n[,epsi])\)。  
   - 实现：`track_frame.Line.xy_to_sn` / `convert.py xy2sn`  
   - 路径：`23-00-38_cut_120m_update/map.mcap`（lap 长约 **2976.875 m**）
2. \(s \in [0, L)\)，\(L=\) 一圈弧长；跨圈帧的 \(s\) 落在同一周期上，可直接比较。  
3. 对 **Lap2∪Lap4 全部建图帧** 按 \(s\) **升序稳定排序**（同 \(s\) 可按 `stamp_ns` 打破平局）。  
4. 得到有序序列 \(\{F_0,\ldots,F_{N-1}\}\) 后，对每个 \(i\) 与  
   \[
   j \in \{i-3,i-2,i-1,i+1,i+2,i+3\} \bmod N
   \]  
   做匹配 + GT 极线三角化（闭环：靠近起终点的帧通过模 \(N\) 与另一端相连——若更稳妥，可用 **弧长圆距离** 取 \(s\)-近邻而非纯索引邻域，见 §5）。

> 说明：文案里「如果第二圈和第三圈同时建图」描述的是**同一套多圈合并逻辑**；本方案**实际采用 Lap2+Lap4 建图、Lap3 query**，避免 query 圈进入地图。

---

## 3. 流水线（拟）

```text
Lap2 camera_r ──┐
                ├─→ 去畸变 + GT 位姿 + (x,y,yaw)→(s,n)
Lap4 camera_r ──┘
                ↓
         按 s 排序（闭环）
                ↓
    对 i 与 i±1,±2,±3：局部描述 mutual NN 匹配（不用 LightGlue）
                ↓
         GT 极线 + 三角化 → points3D
                ↓
         全局描述库（全体建图帧）
                ↓
Lap3 camera_r query → top-5 → mutual NN → PnP（全程 MPS）
                ↓
     metrics (A/B) + err_vs_time + traj_vs_gt
```

> **匹配器澄清**：本方案与 `sfm_hfnet.py` 一致——局部匹配是 **mutual NN + ratio test**，**不是** SuperGlue / LightGlue。旧 `sfm_experiment` 里的 LightGlue 路径不在此实验内。

### 3.1 HF-Net + MPS（硬性）

HF-Net 论文是 **单一网络** 同时出局部特征 + 全局描述；仓库里有两档后端（见 `models/README_HFNET.md`）：

| `BACKEND` | 是什么 | 匹配 |
|---|---|---|
| **`hfnet_onnx`** | **真 HF-Net**（`models/hfnet.onnx`，一体局部+全局） | mutual NN |
| **`sp_nv_nn`** | **代理栈**：分开的 SuperPoint + NetVLAD，流水线形状同论文，**不是**单体 MobileNet | mutual NN |

| 项 | 要求 |
|---|---|
| 模块 | 复用 `sfm_hfnet.py` + 现有 HF-Net notebook 的建图 / localize / metrics 路径 |
| 优先后端 | 有 `models/hfnet.onnx` 时用 **`hfnet_onnx`**；没有则暂用 **`sp_nv_nn`**，并在 `run_meta.json` 标明 `proxy=true` |
| 禁止 | **LightGlue / SuperGlue**（不作为本实验匹配器；勿与「HF-Net」混写） |
| 设备 | notebook 启动时 `device = sfm.pick_device()`，**断言 `device == "mps"`**；特征与全局描述上 MPS |
| 性能意图 | 多圈帧量与边数上升，靠 MPS + NN（相对 LightGlue）压墙钟；OpenCV PnP / 三角化仍可在 CPU |
| 对照 | 与单圈 `outputs_hfnet/map_r` 在**同一 backend + 同一 MPS** 下对比 |

输出根目录建议：`outputs_hfnet_multi_lap_s/`（与 `outputs_hfnet/` 并列）。

---

## 4. 与单圈时间邻域方案的差异

| | 现方案（`sfm_hfnet_*`） | 本方案 |
|---|---|---|
| 特征栈 | HF-Net（优先 onnx / 否则 `sp_nv_nn` 代理） | **相同；匹配 = NN，非 LightGlue** |
| 设备 | MPS（已跑通） | **相同：强制 MPS** |
| 建图数据 | 单圈（如 Lap2） | Lap2 + Lap4 |
| 帧序 | 时间戳 | **racing-line \(s\)** |
| 匹配邻域 | 时间 \(i\pm1,\pm2\) | \(s\)-序 \(i\pm1,\pm2,\pm3\) |
| 跨圈同位置 | 无 | 可成为邻居并三角化 |
| Query | Lap3 | Lap3（相同） |
| 评测 | A/B + 轨迹 | 相同，并与「仅 Lap2 建图」对照 |

---

## 5. 实现细节（review 点）

### 5.1 \(s,n\) 投影

- 输入：每帧 `gt_x, gt_y, gt_yaw, stamp_ns`  
- 输出：`s, n, epsi`（racing line）；可选同时算 pit 线，但建图邻域**只用 track \(s\)**  
- 序列投影：`xy_to_sn(..., t_ns=stamps)`，利用局部窗加速（与车端一致）

### 5.2 排序与邻域

**方案 A（索引邻域，简单）**  
按 \(s\) 排序后取 \(i\pm1,\pm2,\pm3\)（模 \(N\)）。  
风险：两圈采样密度不均时，索引 ±3 对应的 \(\Delta s\) 可能忽大忽小。

**方案 B（弧长邻域，推荐写进实现）**  
对每个 \(i\)，在圆距离  
\[
d_L(s_i,s_j)=\min\bigl(|s_i-s_j|,\; L-|s_i-s_j|\bigr)
\]  
下取最近的 \(K=6\) 个异帧，或取 \(d_L < \tau_s\)（如 15–25 m）内最多 6 帧。  
更贴近「沿赛道前后各约 3 个采样」。

Review 倾向：**实现以方案 B 为主，方案 A 作消融。** 文档标题中的「i±1..±3」对应排序后的默认行为；若采样近似均匀，A≈B。

### 5.3 几何门槛（沿用 / 微调）

- 最小基线 \(\|t\|\)：跨圈同 \(s\)、不同 \(n\)（左右线）时基线可能偏横向，仍可三角化；过近（如 \(\|t\|<0.5\,\mathrm{m}\)）跳过  
- 极线阈值、歧义过滤、PnP `min_inl`：与现 hfnet 实验对齐，便于对比  
- 同一物理灯点可能被两圈各建一次：可接受为重复点，或后续按空间体素合并（非首版必做）

### 5.4 NetVLAD 库

- 库 = Lap2∪Lap4 全部关键帧全局描述  
- Lap3 query top-5：可能命中「同弯道另一圈」帧，属预期行为

### 5.5 输出目录（拟）

```text
3d-map-test/outputs_hfnet_multi_lap_s/
  map_r_lap2+4/          # map_bundle, points3D, netvlad, frames_with_s.csv
  loc_lap3_r/            # errors.csv, submission_xy.csv, plots
  metrics_official.csv
  run_meta.json          # backend, device=mps, stride, neighbor_mode
```

`frames_with_s.csv`：`stamp_ns, lap, s, n, epsi, x, y, yaw` 便于检查排序是否合理。  
`run_meta.json`：记录 `BACKEND`、`device=mps`、邻域模式，保证与单圈 HF-Net 跑次可复现对照。

### 5.6 MPS 使用约定

- 启动 cell：打印并校验 `sfm.pick_device()` → `mps`；失败则 `raise RuntimeError`（多圈实验不以慢 CPU 冒充可跑通）。  
- SP / NetVLAD /（若用）HF-Net ONNX Runtime EP：优先 GPU/MPS 路径；与现 `sfm_hfnet` 加载方式一致。  
- 批量特征缓存写盘后，定位阶段同样走 MPS，避免「建图 MPS、query CPU」的半套加速。

### 5.7 评测（与现 notebook 一致）

对 Lap3 成功位姿：

| 指标 | 来源 |
|---|---|
| Option A Position / lat / lon RMSE、Yaw RMSE | map \(x,y\) |
| Option B \(s,n\) RMSE | 同一 `map.mcap` + `track_frame` |
| within 1.0 m [%] | — |
| Runtime / frame | localize 计时 |
| 图 | `err_vs_time.png`、`traj_vs_gt.png` |

对照：**同一 query（Lap3）+ 同一超参** 下，「仅 Lap2 时间邻域地图」vs「Lap2+4 的 \(s\)-邻域地图」。

---

## 6. 圈次与数据（`laps.json`）

| Lap | 角色 | duration | path |
|---|---|---|---|
| 2 | 建图 | ~167.6 s | ~2993 m |
| 3 | **Query** | ~123.6 s | ~3024 m |
| 4 | 建图 | ~101.3 s | ~2979 m |

帧量（stride=1，~10 Hz）：Lap2≈1600–1700，Lap4≈1000–1100 → 建图合计约 **2.6k–2.8k** 帧；\(s\)-序匹配边约为时间邻域方案的数倍，需注意耗时（可先 `MAP_STRIDE=2` 烟测）。

---

## 7. 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| \(s\) 投影误差 | 进站 / 大横向偏移时 \(n\) 大，\(s\) 仍可用但邻域语义变差 | 可选：丢弃 \(\|n\|>n_{\max}\) 的建图帧 |
| 闭环模运算 | 起终点索引邻域可能跨大空隙 | 优先弧长圆距离邻域 |
| 重复 / 矛盾观测 | 两圈外观微差导致错配 | 极线 + 重投影；歧义过滤 |
| 算力 | 帧数×窗宽上升 | **强制 MPS** + stride + 每帧最多 6 条边 |
| MPS 不可用 | 非 Apple / 驱动异常 | 硬失败并提示；不默认降级 CPU 跑全量 |
| 与 query 泄漏 | 若误把 Lap3 加入建图会偏乐观 | **硬约束：建图仅 2+4** |

---

## 8. 建议实施顺序（通过 review 后）

1. 新建 notebook（建议名 `sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb`）：首 cell 校验 **MPS + HF-Net backend**。  
2. 写 `frames_with_s.csv` 烟测：Lap2+4 投影、排序、可视化 \(s\) 直方图 / 沿轨着色。  
3. 实现 `build_map_s_neighbors(...)`（弧长邻域 + 可选索引 ±3），特征走 `sfm_hfnet` / MPS。  
4. 全量或 stride=2 建 `map_r_lap2+4` → `outputs_hfnet_multi_lap_s/`。  
5. Lap3 query + metrics/轨迹（复用 `compute_camloc_metrics` / hfnet 绘图）。  
6. 与单圈 `outputs_hfnet/map_r`（同 backend、同 MPS）结果表对照。

---

## 9. Review 清单（请确认）

- [x] **必须使用 HF-Net**（优先 `hfnet_onnx`；无权重时才用 `sp_nv_nn` 代理）  
- [x] **局部匹配 = mutual NN**（明确 **不用 LightGlue**）  
- [x] **必须启用 MPS**（启动校验；建图 + query 特征均上 MPS）  
- [ ] 建图圈确认为 **Lap2 + Lap4**（不是 2+3）  
- [ ] Query 确认为 **Lap3 only**  
- [ ] 邻域用 **排序后 i±1..±3**，是否改为 / 同时提供 **弧长圆距离 Top-6**  
- [ ] 首版是否丢弃 \(\|n\|>n_{\max}\) 的帧  
- [ ] 烟测 stride / 全圈策略  

实现入口：`sfm_hfnet_r_lap2_lap4_s_order_lap3_loc.ipynb` → `outputs_hfnet_multi_lap_s/`。
