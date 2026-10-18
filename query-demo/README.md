# query-demo (EKF)

Design spec: [`QUERY_DEMO_EKF.md`](QUERY_DEMO_EKF.md)

## Notebooks

| Notebook | Lap | Vision loc |
|----------|-----|------------|
| [`lap3_ekf_query_demo.ipynb`](lap3_ekf_query_demo.ipynb) | Lap 3 | `loc_lap3_*` (multi-lap notebook) |
| [`lap5_ekf_query_demo.ipynb`](lap5_ekf_query_demo.ipynb) | Lap 5 | `loc_lap5_*` (run `ensure_lap_query.py --lap 5` first) |

Kernel: `pixel-racers`. Maps are all Lap2+4 (`map_*_lap2+4`).

## Lap5 prerequisite query

```bash
cd query-demo
python ensure_lap_query.py --lap 5    # reuse maps, run r/fr/rl query only (slow)
```

## Workflow (same for both notebooks)

1. Write `config/process_noise.yaml` (MADσ)
2. **Mode A** `se_predict` → `outputs/lap{N}/se_predict/`
3. **Mode B** `ekf_cam` → `outputs/lap{N}/ekf_cam/`
4. Compare `se_predict` / `fused_se2` / `ekf_cam` → `outputs/lap{N}/compare/`

Evaluation uses clipped GT: `outputs/lap{N}/gt_lap{N}.csv`.

Core libraries: [`query_demo_ekf.py`](query_demo_ekf.py), [`ensure_lap_query.py`](ensure_lap_query.py)
