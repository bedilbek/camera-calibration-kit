"""Live stereo processing with a bounded latest-result mailbox, independent of widgets."""

from pathlib import Path
from threading import Lock
from time import monotonic

import cv2
import numpy as np
from PySide6.QtCore import QThread

from .geometry import Rig, reconstruct
from .offline import detect_tags, load_intrinsic, tag_detector


def load_live_rig(pose_path, left_intrinsic=None, right_intrinsic=None):
    """Optional intrinsic imports must match the lenses used to solve this pose."""
    source = Path(pose_path)
    rig = Rig.load(source / "calibration.json" if source.is_dir() else source)
    if rig.metadata.get("synthetic"):
        raise ValueError("Live capture requires a REAL calibration, not a synthetic rig")
    for index, path in enumerate((left_intrinsic, right_intrinsic), 1):
        if not path:
            continue  # Sequential export already embeds both sets of intrinsics.
        K, d, size, _, _ = load_intrinsic(path)
        if size != rig.size:
            raise ValueError(f"Camera {index} intrinsic resolution differs from the saved rig")
        if not (
            np.allclose(K, getattr(rig, f"K{index}"), rtol=1e-9, atol=1e-9)
            and d.size == getattr(rig, f"d{index}").size
            and np.allclose(d.ravel(), getattr(rig, f"d{index}").ravel(), rtol=1e-9, atol=1e-9)
        ):
            raise ValueError(f"Camera {index} intrinsics differ; recompute the sequential pose")
    return rig


def process_pair(rig, images, ids, detector=None):
    """One fast reconstruction per pair; no per-frame Monte Carlo or temporal smoothing."""
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("Choose two different marker IDs")
    if len(images) != 2 or any(im.shape[1::-1] != rig.size for im in images):
        raise ValueError(f"Both streams must match calibrated resolution {rig.size}")
    found = []
    for image in images:
        try:
            found.append(detect_tags(image, detector=detector))
        except ValueError as exc:
            return {"result": None, "centers": found, "status": str(exc)}
    missing = [sorted(set(ids) - view.keys()) for view in found]
    if any(missing):
        return {
            "result": None,
            "centers": found,
            "status": f"Missing IDs — left: {missing[0]}, right: {missing[1]}",
        }
    points = [np.asarray([view[i] for i in ids]) for view in found]
    try:
        result = reconstruct(rig, *points)
    except ValueError as exc:
        return {"result": None, "centers": found, "status": str(exc)}
    # Keep the diagnostics, but do not present an unreliable length as valid tracking.
    status = "; ".join(result["warnings"])
    return {
        "result": result if not status else None,
        "centers": found,
        "status": status or "Tracking",
    }


class LiveWorker(QThread):
    """Own both devices and detection. UI polls one replaceable packet, never a frame queue."""

    def __init__(self, indices, rig, ids):
        super().__init__()
        self.indices, self.rig, self.ids = indices, rig, ids
        self.lock = Lock()
        self.packet = None

    def publish(self, packet):
        with self.lock:
            self.packet = packet

    def take(self):
        with self.lock:
            packet, self.packet = self.packet, None
        return packet

    def run(self):
        caps = []
        try:
            if len(self.indices) != 2 or self.indices[0] == self.indices[1]:
                raise ValueError("Select two different camera indices")
            detector = tag_detector()
            for index in self.indices:
                cap = cv2.VideoCapture(index)
                caps.append(cap)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open camera {index}; check permissions/other apps")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.rig.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.rig.size[1])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Backend may ignore this request.
            previous = monotonic()
            while not self.isInterruptionRequested():
                stamps, images = [], []
                for cap in caps:
                    if not cap.grab():
                        raise RuntimeError("Camera disconnected or frame grab failed")
                    stamps.append(monotonic())
                for cap in caps:
                    ok, image = cap.retrieve()
                    if not ok or image is None:
                        raise RuntimeError("Camera frame retrieval failed")
                    images.append(image)
                packet = process_pair(self.rig, images, self.ids, detector)
                now = monotonic()
                packet.update(
                    images=images,
                    captured_at=min(stamps),
                    processed_at=now,
                    fps=1 / max(now - previous, 1e-6),
                    host_gap_ms=1000 * abs(stamps[1] - stamps[0]),
                )
                previous = now
                self.publish(packet)
        except Exception as exc:
            self.publish({"error": str(exc)})
        finally:
            for cap in caps:
                cap.release()
