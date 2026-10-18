# query-demo（EKF）

设计文档：[`QUERY_DEMO_EKF.md`](QUERY_DEMO_EKF.md)

## Notebooks

| Notebook | 圈 | 视觉 loc |
|----------|----|----------|
| [`lap3_ekf_query_demo.ipynb`](lap3_ekf_query_demo.ipynb) | Lap 3 | `loc_lap3_*`（multi-lap notebook） |
| [`lap5_ekf_query_demo.ipynb`](lap5_ekf_query_demo.ipynb) | Lap 5 | `loc_lap5_*`（先 `ensure_lap_query.py --lap 5`） |

Kernel：`pixel-racers`。地图均为 Lap2+4（`map_*_lap2+4`）。

## Lap5 前置 query

```bash
cd query-demo
python ensure_lap_query.py --lap 5    # 复用地图，只跑 r/fr/rl query（较慢）
```

## 流程（两 notebook 相同）

1. 写 `config/process_noise.yaml`（MADσ）
2. **Mode A** `se_predict` → `outputs/lap{N}/se_predict/`
3. **Mode B** `ekf_cam` → `outputs/lap{N}/ekf_cam/`
4. 对比 `se_predict` / `fused_se2` / `ekf_cam` → `outputs/lap{N}/compare/`

评测用裁剪 GT：`outputs/lap{N}/gt_lap{N}.csv`。

核心库：[`query_demo_ekf.py`](query_demo_ekf.py)、[`ensure_lap_query.py`](ensure_lap_query.py)
