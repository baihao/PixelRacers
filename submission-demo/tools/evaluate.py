#!/usr/bin/env python3
"""Score a camera-localization submission against the ground truth.

    python3 evaluate.py gt.csv submission.csv

submission.csv: timestamp_ns,x,y,yaw  (header line required; pose of constructor0/base_link in frame
map at camera time timestamp_ns -- use the header.stamp of the image(s) the pose was computed from).

GT is interpolated linearly at each submitted timestamp (yaw on the unwrapped angle). Errors are split
into the GT body frame: longitudinal (along the car heading) and lateral (across it).
"""
import sys
import numpy as np

MAX_GT_GAP_NS = 50_000_000        # GT gap around a submitted stamp larger than this -> stamp ignored
BIN_NS = 1_000_000_000            # coverage: share of 1 s bins of the GT window with >= 1 prediction
MIN_COVERAGE = 0.90               # below this the submission is not ranked


def load(path, cols):
    d = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None)
    return {c: np.asarray(d[c], dtype=np.int64 if c == 'timestamp_ns' else float) for c in cols}


def score_submission(gt_path, sub_path):
    """Official metrics as a dict (same rules as printed report)."""
    gt = load(gt_path, ["timestamp_ns", "x", "y", "yaw"])
    sb = load(sub_path, ["timestamp_ns", "x", "y", "yaw"])
    o = np.argsort(gt['timestamp_ns']); gt = {k: v[o] for k, v in gt.items()}
    o = np.argsort(sb['timestamp_ns']); sb = {k: v[o] for k, v in sb.items()}
    tg, ts = gt['timestamp_ns'], sb['timestamp_ns']

    i = np.searchsorted(tg, ts)
    ok = (i > 0) & (i < len(tg))
    i = np.clip(i, 1, len(tg) - 1)
    ok &= (tg[i] - tg[i - 1]) <= MAX_GT_GAP_NS
    n_sub, ts, i = len(ts), ts[ok], i[ok]
    sx, sy, syaw = sb['x'][ok], sb['y'][ok], sb['yaw'][ok]
    a = (ts - tg[i - 1]) / (tg[i] - tg[i - 1])
    yaw_u = np.unwrap(gt['yaw'])
    gx = gt['x'][i - 1] + a * (gt['x'][i] - gt['x'][i - 1])
    gy = gt['y'][i - 1] + a * (gt['y'][i] - gt['y'][i - 1])
    gyaw = yaw_u[i - 1] + a * (yaw_u[i] - yaw_u[i - 1])

    dx, dy = sx - gx, sy - gy
    lon = np.cos(gyaw) * dx + np.sin(gyaw) * dy
    lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
    pos = np.hypot(dx, dy)
    dyaw = np.degrees(np.angle(np.exp(1j * (syaw - gyaw))))

    bins = np.arange(tg[0], tg[-1], BIN_NS)
    covered = np.unique(np.searchsorted(bins, ts, side='right') - 1)
    coverage = len(covered) / max(len(bins), 1)
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")

    out = {
        "poses_submitted": int(n_sub),
        "poses_evaluated": int(len(ts)),
        "coverage_pct": float(coverage * 100),
        "min_coverage_pct": float(MIN_COVERAGE * 100),
        "position_rmse_m": rms(pos),
        "position_median_m": float(np.median(pos)) if len(pos) else float("nan"),
        "position_p95_m": float(np.percentile(pos, 95)) if len(pos) else float("nan"),
        "lateral_rmse_m": rms(lat),
        "longitudinal_rmse_m": rms(lon),
        "yaw_rmse_deg": rms(dyaw),
        "within_0.25m_pct": float(np.mean(pos < 0.25) * 100) if len(pos) else float("nan"),
        "within_0.50m_pct": float(np.mean(pos < 0.50) * 100) if len(pos) else float("nan"),
        "within_1.00m_pct": float(np.mean(pos < 1.00) * 100) if len(pos) else float("nan"),
        "ranked": bool(coverage >= MIN_COVERAGE),
    }
    return out


def format_report(metrics: dict) -> str:
    m = metrics
    lines = [
        f"poses submitted        {m['poses_submitted']}   evaluated {m['poses_evaluated']}",
        f"coverage (1 s bins)    {m['coverage_pct']:.1f} %   "
        f"(>= {m['min_coverage_pct']:.0f} % to be ranked)",
        f"position RMSE  [m]     {m['position_rmse_m']:.3f}   <- SCORE (lower is better)",
        f"position median/p95    {m['position_median_m']:.3f} / {m['position_p95_m']:.3f}",
        f"lateral RMSE   [m]     {m['lateral_rmse_m']:.3f}",
        f"longitudinal RMSE [m]  {m['longitudinal_rmse_m']:.3f}",
        f"yaw RMSE     [deg]     {m['yaw_rmse_deg']:.3f}",
        f"within 0.25 m          {m['within_0.25m_pct']:.1f} %",
        f"within 0.50 m          {m['within_0.50m_pct']:.1f} %",
        f"within 1.00 m          {m['within_1.00m_pct']:.1f} %",
    ]
    if not m["ranked"]:
        lines.append("NOT RANKED: coverage below threshold")
    return "\n".join(lines)


def main(gt_path, sub_path):
    print(format_report(score_submission(gt_path, sub_path)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
