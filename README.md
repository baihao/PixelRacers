# PixelRacers

Constructor Labs 赛道相机定位实验环境。

## 环境：`pixel-racers`

```bash
cd /Users/hbai/SchubKI/PixelRacers
python3.12 -m venv pixel-racers
source pixel-racers/bin/activate
python -m pip install -U pip
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m ipykernel install --user --name pixel-racers --display-name "pixel-racers"
```

## 组件

| 路径 | 说明 |
|------|------|
| `camloc_play.py` | 用 mcap / mcap-ros2-support 读相机 JPEG（无需完整 ROS 2） |
| `third_party/Hierarchical-Localization` | **git submodule**（与 `v2.0` 同上游 / 同 commit） |
| `test/play_mcap.ipynb` | 快速查看 FL 等相机流 |

默认 bag：

```text
../c0_2026-10-13_22-53-42_camloc/c0_2026-10-13_22-53-42_camloc_0.mcap
```

## 快速试播

```bash
source pixel-racers/bin/activate
python camloc_play.py --list-topics
python camloc_play.py --camera fl --first
# GUI（需桌面）:
# python camloc_play.py --camera fl --play --duration 5
```

## Submodule

```bash
git submodule update --init --recursive
# 上游: https://github.com/cvg/Hierarchical-Localization.git
```
