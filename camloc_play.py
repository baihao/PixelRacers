"""Play / iterate Constructor Labs camloc mcap (no full ROS 2 Desktop).

Uses ``mcap`` + ``mcap-ros2-support``. Default bag lives next to this repo under
``../c0_2026-10-13_22-53-42_camloc/``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

DEFAULT_MCAP = (
    Path(__file__).resolve().parents[1]
    / "c0_2026-10-13_22-53-42_camloc"
    / "c0_2026-10-13_22-53-42_camloc_0.mcap"
)

CAMERAS = ("fl", "fr", "cl", "cr", "r", "rl", "rr")


def camera_topic(cam: str) -> str:
    return f"/constructor0/sensor/camera_{cam}/compressed_image"


def stamp_ns_from_msg(msg) -> int:
    return int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)


def decode_jpeg(msg) -> np.ndarray:
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return img


@dataclass
class ImageFrame:
    camera: str
    stamp_ns: int
    log_time_ns: int
    bgr: np.ndarray


def iter_camera_frames(
    mcap_path: Path | str,
    cameras: Sequence[str] = ("fl",),
    *,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> Iterator[ImageFrame]:
    """Yield decoded JPEG frames (bag / log-time order)."""
    topics = [camera_topic(c) for c in cameras]
    cam_by_topic = {camera_topic(c): c for c in cameras}
    n = 0
    log_start = None if start_ns is None else max(0, int(start_ns) - 2_000_000_000)
    log_end = None if end_ns is None else int(end_ns) + 2_000_000_000
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, msg in reader.iter_decoded_messages(
            topics=topics,
            start_time=log_start,
            end_time=log_end,
        ):
            stamp = stamp_ns_from_msg(msg)
            if start_ns is not None and stamp < start_ns:
                continue
            if end_ns is not None and stamp > end_ns:
                continue
            yield ImageFrame(
                camera=cam_by_topic[channel.topic],
                stamp_ns=stamp,
                log_time_ns=int(message.log_time),
                bgr=decode_jpeg(msg),
            )
            n += 1
            if max_frames is not None and n >= max_frames:
                break


def load_first_image(
    mcap_path: Path | str, camera: str = "fl"
) -> Tuple[int, np.ndarray]:
    for fr in iter_camera_frames(mcap_path, cameras=(camera,), max_frames=1):
        return fr.stamp_ns, fr.bgr
    raise RuntimeError(f"no images for camera={camera}")


def list_topics(mcap_path: Path | str) -> List[str]:
    with open(mcap_path, "rb") as f:
        summary = make_reader(f).get_summary()
        if summary is None or summary.statistics is None:
            return []
        return sorted({ch.topic for ch in summary.channels.values()})


def bag_summary(mcap_path: Path | str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, _message, _msg in reader.iter_decoded_messages():
            counts[channel.topic] = counts.get(channel.topic, 0) + 1
    return counts


def play_camera(
    mcap_path: Path | str,
    camera: str = "fl",
    *,
    start_ns: Optional[int] = None,
    duration_s: float = 5.0,
    speed: float = 1.0,
    max_frames: Optional[int] = None,
    window_name: str = "pixel-racers",
) -> int:
    """OpenCV playback paced by stamp deltas."""
    end_ns = None if start_ns is None else start_ns + int(duration_s * 1e9)
    prev_stamp: Optional[int] = None
    shown = 0
    for fr in iter_camera_frames(
        mcap_path,
        cameras=(camera,),
        start_ns=start_ns,
        end_ns=end_ns,
        max_frames=max_frames,
    ):
        if prev_stamp is not None and speed > 0:
            dt = (fr.stamp_ns - prev_stamp) / 1e9 / speed
            if dt > 0:
                time.sleep(min(dt, 0.5))
        prev_stamp = fr.stamp_ns
        canvas = fr.bgr.copy()
        cv2.putText(
            canvas,
            f"{fr.camera}  t={fr.stamp_ns}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, canvas)
        shown += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    return shown


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcap", type=Path, default=DEFAULT_MCAP)
    p.add_argument("--camera", default="fl", choices=CAMERAS)
    p.add_argument("--list-topics", action="store_true")
    p.add_argument("--first", action="store_true", help="decode first frame and print shape")
    p.add_argument("--play", action="store_true")
    p.add_argument("--duration", type=float, default=5.0)
    args = p.parse_args()

    if not args.mcap.is_file():
        raise SystemExit(f"mcap not found: {args.mcap}")

    if args.list_topics:
        for t in list_topics(args.mcap):
            print(t)
    elif args.play:
        n = play_camera(args.mcap, camera=args.camera, duration_s=args.duration)
        print(f"shown {n} frames")
    else:
        # default: first frame
        stamp, bgr = load_first_image(args.mcap, camera=args.camera)
        print(f"camera={args.camera} stamp_ns={stamp} shape={bgr.shape}")
