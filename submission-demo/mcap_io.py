"""mcap image iteration + JPEG decode (self-contained, from camloc_play.py).

Yields decoded camera frames from a ROS2 mcap in bag/log-time order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

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
    cameras: Sequence[str] = ("fr",),
    *,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> Iterator[ImageFrame]:
    """Yield decoded JPEG frames (bag / log-time order) filtered by header stamp."""
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


def camera_stamps(
    mcap_path: Path | str,
    camera: str,
    *,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
) -> List[int]:
    """Header stamps for one camera in the window WITHOUT decoding JPEGs.

    Much cheaper than iter_camera_frames since it never calls cv2.imdecode;
    used for timeline-only needs (e.g. motion-only t_last sampling).
    """
    topic = camera_topic(camera)
    out: List[int] = []
    log_start = None if start_ns is None else max(0, int(start_ns) - 2_000_000_000)
    log_end = None if end_ns is None else int(end_ns) + 2_000_000_000
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, _channel, _message, msg in reader.iter_decoded_messages(
            topics=[topic], start_time=log_start, end_time=log_end
        ):
            stamp = stamp_ns_from_msg(msg)
            if start_ns is not None and stamp < start_ns:
                continue
            if end_ns is not None and stamp > end_ns:
                continue
            out.append(stamp)
    return out


def mcap_time_bounds(mcap_path: Path | str, camera: str = "fr") -> Tuple[int, int]:
    """First/last camera header stamp across the whole bag (for whole-mcap runs)."""
    lo: Optional[int] = None
    hi: Optional[int] = None
    for fr in iter_camera_frames(mcap_path, cameras=(camera,)):
        if lo is None or fr.stamp_ns < lo:
            lo = fr.stamp_ns
        if hi is None or fr.stamp_ns > hi:
            hi = fr.stamp_ns
    if lo is None or hi is None:
        raise RuntimeError(f"no camera_{camera} frames in {mcap_path}")
    return int(lo), int(hi)
