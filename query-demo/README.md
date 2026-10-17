# query-demo（Lap3 EKF）

设计文档：[`QUERY_DEMO_EKF.md`](QUERY_DEMO_EKF.md)

## Notebook

[`lap3_ekf_query_demo.ipynb`](lap3_ekf_query_demo.ipynb) — Kernel：`pixel-racers`

流程：

1. 从 `state-error-estimation/se_step_residual_noise.json` 写 `config/process_noise.yaml`（MADσ）
2. **Mode A** 纯 SE 航位 → `outputs/lap3/se_predict/`
3. **Mode B** EKF + r/fr/rl sync/外推/se2 + \(R(s)\) → `outputs/lap3/ekf_cam/`
4. 对比 metrics / 轨迹 / err_vs_s → `outputs/lap3/compare/`

视觉观测默认复用 `3d-map-test/outputs_hfnet_multi_lap_s/loc_lap3_{r,fr,rl}`（与 multi-lap notebook 同一 SfM query）。

评测使用裁剪后的 `outputs/lap3/gt_lap3.csv`（整段 GT 会导致 coverage 虚低）。

核心库：[`query_demo_ekf.py`](query_demo_ekf.py)
