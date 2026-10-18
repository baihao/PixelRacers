#!/usr/bin/env python3
"""Extract calibrations, track bounds, and car state from the mcap bag.

    python tools/extract_assets.py c0_..._0.mcap extracted/

Writes:
  extracted/camera_info.json
  extracted/tf_static.json
  extracted/track_bounds.json
  extracted/state_estimation.csv   # pose + twist + accel when available
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


CAMERAS = ("fl", "fr", "cl", "cr", "r", "rl", "rr")


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 10**9 + int(stamp.nanosec)


def _quat_yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _camera_info_dict(msg) -> dict:
    return {
        "frame_id": msg.header.frame_id,
        "width": int(msg.width),
        "height": int(msg.height),
        "distortion_model": msg.distortion_model,
        "D": list(msg.d),
        "K": list(msg.k),
        "R": list(msg.r),
        "P": list(msg.p),
    }


def _transform_dict(t) -> dict:
    p, q = t.transform.translation, t.transform.rotation
    return {
        "parent": t.header.frame_id,
        "child": t.child_frame_id,
        "translation": {"x": p.x, "y": p.y, "z": p.z},
        "rotation": {"x": q.x, "y": q.y, "z": q.z, "w": q.w},
        "yaw": _quat_yaw(q),
    }


def _marker_polyline(marker) -> dict:
    pts = [{"x": p.x, "y": p.y, "z": p.z} for p in marker.points]
    return {
        "ns": marker.ns,
        "id": int(marker.id),
        "frame_id": marker.header.frame_id,
        "type": int(marker.type),
        "points": pts,
        "n_points": len(pts),
    }


def main(bag: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    camera_info: dict[str, dict] = {}
    tf_static: dict[str, dict] = {}
    track_bounds: dict[str, dict] = {}
    state_rows: list[list] = []

    TWIST_TOPIC = "/constructor0/validation/twist"
    ACCEL_TOPIC = "/constructor0/validation/accel"

    topics = [
        *[f"/constructor0/sensor/camera_{c}/camera_info" for c in CAMERAS],
        "/tf_static",
        "/constructor0/track_bounds",
        "/constructor0/state_estimation",
        TWIST_TOPIC,
        ACCEL_TOPIC,
    ]

    validation_twist: list[list] = []
    validation_accel: list[list] = []

    with open(bag, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, msg in reader.iter_decoded_messages(topics=topics):
            name = channel.topic

            if name.endswith("/camera_info"):
                cam = name.split("camera_")[1].split("/")[0]
                if cam not in camera_info:
                    camera_info[cam] = _camera_info_dict(msg)
                continue

            if name == "/tf_static":
                for t in msg.transforms:
                    key = f"{t.header.frame_id}->{t.child_frame_id}"
                    if key not in tf_static:
                        tf_static[key] = _transform_dict(t)
                continue

            if name == "/constructor0/track_bounds":
                for marker in msg.markers:
                    key = marker.ns or f"id_{marker.id}"
                    if key not in track_bounds and marker.points:
                        track_bounds[key] = _marker_polyline(marker)
                continue

            if name == TWIST_TOPIC:
                ts = _stamp_ns(msg.header.stamp)
                tw = msg.twist
                validation_twist.append(
                    [
                        ts,
                        f"{tw.linear.x:.6f}",
                        f"{tw.linear.y:.6f}",
                        f"{tw.linear.z:.6f}",
                        f"{tw.angular.x:.6f}",
                        f"{tw.angular.y:.6f}",
                        f"{tw.angular.z:.6f}",
                    ]
                )
                continue

            if name == ACCEL_TOPIC:
                ts = _stamp_ns(msg.header.stamp)
                a = msg.accel
                validation_accel.append(
                    [
                        ts,
                        f"{a.linear.x:.6f}",
                        f"{a.linear.y:.6f}",
                        f"{a.linear.z:.6f}",
                        f"{a.angular.x:.6f}",
                        f"{a.angular.y:.6f}",
                        f"{a.angular.z:.6f}",
                    ]
                )
                continue

            if name == "/constructor0/state_estimation":
                # sd_localization_msgs/StateEstimation — flat fields (not geometry_msgs Pose).
                stamp = msg.header.stamp
                ts = _stamp_ns(stamp)
                state_rows.append(
                    [
                        ts,
                        f"{msg.x_m:.4f}",
                        f"{msg.y_m:.4f}",
                        f"{msg.z_m:.4f}",
                        f"{msg.yaw_rad:.6f}",
                        f"{msg.v_mps:.6f}",
                        f"{msg.vx_mps:.6f}",
                        f"{msg.vy_mps:.6f}",
                        f"{msg.vz_mps:.6f}",
                        f"{msg.wx_radps:.6f}",
                        f"{msg.wy_radps:.6f}",
                        f"{msg.wz_radps:.6f}",
                        f"{msg.ax_mps2:.6f}",
                        f"{msg.ay_mps2:.6f}",
                        f"{msg.az_mps2:.6f}",
                        f"{msg.delta_wheel_rad:.6f}",
                        int(msg.gear),
                        f"{msg.yaw_vel_rad:.6f}",
                    ]
                )

    (out / "camera_info.json").write_text(
        json.dumps(camera_info, indent=2) + "\n", encoding="utf-8"
    )
    (out / "tf_static.json").write_text(
        json.dumps(tf_static, indent=2) + "\n", encoding="utf-8"
    )
    (out / "track_bounds.json").write_text(
        json.dumps(track_bounds, indent=2) + "\n", encoding="utf-8"
    )

    motion_source = "state_estimation"
    if not state_rows and validation_twist:
        validation_twist.sort(key=lambda r: int(r[0]))
        validation_accel.sort(key=lambda r: int(r[0]))
        accel_ts = [int(r[0]) for r in validation_accel]

        def _nearest_accel(ts: int) -> list[str]:
            if not validation_accel:
                return ["0", "0", "0", "0", "0", "0"]
            j = min(
                range(len(accel_ts)),
                key=lambda k: abs(accel_ts[k] - ts),
            )
            return validation_accel[j][1:]

        for row in validation_twist:
            ts, vx, vy, vz, wx, wy, wz = row
            ax, ay, az, _aax, _aay, _aaz = _nearest_accel(int(ts))
            v_mps = math.hypot(float(vx), float(vy))
            state_rows.append(
                [
                    ts,
                    "",  # x — no pose on test validation topics
                    "",
                    "",
                    "",
                    f"{v_mps:.6f}",
                    vx,
                    vy,
                    vz,
                    wx,
                    wy,
                    wz,
                    ax,
                    ay,
                    az,
                    "",
                    "",
                    wz,
                ]
            )
        motion_source = "validation_twist_accel"
        print(
            f"validation twist/accel -> state_estimation rows: {len(state_rows)} "
            f"(twist={len(validation_twist)} accel={len(validation_accel)})"
        )

    (out / "motion_meta.json").write_text(
        json.dumps(
            {
                "source": motion_source,
                "n_state_rows": len(state_rows),
                "validation_twist": len(validation_twist),
                "validation_accel": len(validation_accel),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    state_path = out / "state_estimation.csv"
    with state_path.open("w", newline="", encoding="utf-8") as o:
        w = csv.writer(o)
        w.writerow(
            [
                "timestamp_ns",
                "x",
                "y",
                "z",
                "yaw",
                "v_mps",
                "vx_mps",
                "vy_mps",
                "vz_mps",
                "wx_radps",
                "wy_radps",
                "wz_radps",
                "ax_mps2",
                "ay_mps2",
                "az_mps2",
                "delta_wheel_rad",
                "gear",
                "yaw_vel_rad",
            ]
        )
        w.writerows(state_rows)

    print(f"cameras: {sorted(camera_info)}")
    print(f"tf_static transforms: {len(tf_static)}")
    print(f"track_bounds: {list(track_bounds)} sizes={[track_bounds[k]['n_points'] for k in track_bounds]}")
    print(f"state_estimation rows: {len(state_rows)} -> {state_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
