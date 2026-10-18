"""Plotting: per-option (metrics + trajectory) and summary comparison figures.

Trajectory plots overlay the race-track outline (track_left / track_right from
the extracted track_bounds.json) with GT (eval) and each option's predicted path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

OPTION_COLORS = {
    "motion_only": "#e74c3c",
    "two_cam": "#2980b9",
    "ekf": "#8e44ad",
}
OPTION_LABEL = {
    "motion_only": "motion only",
    "two_cam": "two-camera",
    "ekf": "EKF",
}
METRIC_KEYS = [
    ("pos_rmse_xy_m", "pos RMSE xy [m]"),
    ("pos_rmse_sn_m", "pos RMSE s/n [m]"),
    ("yaw_rmse_deg", "yaw RMSE [deg]"),
    ("within_1m_of_all_%", "within 1 m [%]"),
]


def load_track_outline(track_bounds_json: Path) -> Dict[str, np.ndarray]:
    """track_left / track_right polylines (Nx2) from extracted track_bounds.json."""
    out: Dict[str, np.ndarray] = {}
    p = Path(track_bounds_json)
    if not p.is_file():
        return out
    data = json.loads(p.read_text())
    for key in ("track_left", "track_right"):
        if key in data and data[key].get("points"):
            out[key] = np.array(
                [[pt["x"], pt["y"]] for pt in data[key]["points"]], dtype=float
            )
    return out


def load_track_outline_from_map_mcap(map_mcap: Path) -> Dict[str, np.ndarray]:
    """Fallback for test bags without /constructor0/track_bounds: use track-model map.mcap."""
    import sys

    tools = Path(__file__).resolve().parent / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from track_frame import load_lines  # type: ignore

    lines = load_lines(str(map_mcap))
    out: Dict[str, np.ndarray] = {"racing_line": np.asarray(lines["track"].p, float)}
    if "pit" in lines:
        out["pit_lane"] = np.asarray(lines["pit"].p, float)
    return out


def resolve_track_outline(
    track_bounds_json: Path,
    map_mcap: Path,
    *,
    track_line=None,
) -> Dict[str, np.ndarray]:
    """Racing line + track_left/right for trajectory plots (eval + predict).

    Boundaries: bag ``track_bounds.json`` if present, else
    ``<track-model>/track_bounds.json`` next to ``map.mcap``.
    """
    out: Dict[str, np.ndarray] = {}
    if track_line is not None and hasattr(track_line, "p"):
        out["racing_line"] = np.asarray(track_line.p, float)
    mp = Path(map_mcap)
    if "racing_line" not in out and mp.is_file():
        out.update(load_track_outline_from_map_mcap(mp))

    bounds = load_track_outline(track_bounds_json)
    if not bounds:
        fallback = mp.parent / "track_bounds.json"
        bounds = load_track_outline(fallback)
    for key in ("track_left", "track_right"):
        if key in bounds:
            out[key] = bounds[key]
    return out


def _draw_track(ax, outline: Dict[str, np.ndarray]) -> None:
    if "racing_line" in outline:
        xy = outline["racing_line"]
        ax.plot(
            xy[:, 0], xy[:, 1], color="#bdc3c7", lw=1.2, zorder=1, label="racing line"
        )
    if "pit_lane" in outline:
        xy = outline["pit_lane"]
        ax.plot(xy[:, 0], xy[:, 1], color="#d5dbdb", lw=0.9, ls="--", zorder=1)
    drew_bounds = False
    for key, color in (("track_left", "#7f8c8d"), ("track_right", "#95a5a6")):
        if key in outline:
            xy = outline[key]
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=color,
                lw=1.0,
                zorder=1,
                label="track boundary" if not drew_bounds else "_track_boundary",
            )
            drew_bounds = True


def plot_option(
    *,
    name: str,
    pred: pd.DataFrame,
    gt_at: Optional[pd.DataFrame],
    metrics: Optional[dict],
    outline: Dict[str, np.ndarray],
    out_path: Path,
    show: bool = True,
):
    """One figure per option: metrics bar chart (eval) + trajectory on track."""
    import matplotlib.pyplot as plt

    color = OPTION_COLORS.get(name, "#333333")
    label = OPTION_LABEL.get(name, name)
    has_metrics = metrics is not None

    ncols = 2 if has_metrics else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 6.0))
    if ncols == 1:
        axes = [axes]

    col = 0
    if has_metrics:
        ax = axes[col]
        keys = [(k, lab) for k, lab in METRIC_KEYS if k in metrics and np.isfinite(metrics.get(k, np.nan))]
        vals = [float(metrics[k]) for k, _ in keys]
        labs = [lab for _, lab in keys]
        bars = ax.bar(range(len(vals)), vals, color=color, alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labs, rotation=20, ha="right")
        ax.set_title(f"{label}: official metrics")
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
        col += 1

    ax = axes[col]
    _draw_track(ax, outline)
    if gt_at is not None:
        ax.plot(gt_at["x"], gt_at["y"], color="#27ae60", lw=1.8, label="GT", zorder=2)
    ax.plot(pred["x"], pred["y"], color=color, lw=1.2, label=label, zorder=3)
    if len(pred):
        ax.scatter([pred["x"].iloc[0]], [pred["y"].iloc[0]], c="#f39c12",
                   s=45, zorder=5, label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"{label}: trajectory vs GT" if gt_at is not None else f"{label}: trajectory")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"Option — {label}", y=1.02, fontsize=12)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_summary_metrics(
    mdf: pd.DataFrame,
    *,
    out_path: Path,
    show: bool = True,
):
    """Grouped bar charts comparing options across the four official metrics."""
    import matplotlib.pyplot as plt

    modes = list(mdf["mode"])
    x = np.arange(len(modes))
    keys = [(k, lab) for k, lab in METRIC_KEYS if k in mdf.columns]
    n = len(keys)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    colors = [OPTION_COLORS.get(m, "#333333") for m in modes]
    for i, (k, lab) in enumerate(keys):
        ax = axes[i]
        vals = mdf[k].to_numpy(float)
        bars = ax.bar(x, vals, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([OPTION_LABEL.get(m, m) for m in modes], rotation=15, ha="right")
        ax.set_title(lab)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Summary — official metrics by option", y=1.02, fontsize=12)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_summary_trajectories(
    preds: Dict[str, pd.DataFrame],
    *,
    gt_at: Optional[pd.DataFrame],
    outline: Dict[str, np.ndarray],
    out_path: Path,
    title: str = "Summary — trajectories",
    show: bool = True,
):
    """All option trajectories on one race-track plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_track(ax, outline)
    if gt_at is not None:
        ax.plot(gt_at["x"], gt_at["y"], color="#27ae60", lw=2.0, label="GT", zorder=2)
    for name, pred in preds.items():
        ax.plot(
            pred["x"], pred["y"],
            color=OPTION_COLORS.get(name, "#333333"),
            lw=1.2, alpha=0.9, label=OPTION_LABEL.get(name, name), zorder=3,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
