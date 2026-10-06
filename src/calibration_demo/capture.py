"""One owner thread per capture rig. Software pairing is not exposure synchronization."""

from time import monotonic

import cv2
from PySide6.QtCore import QThread, Signal


class CaptureWorker(QThread):
    frames = Signal(object)
    failure = Signal(str)

    def __init__(self, indices, size):
        super().__init__()
        self.indices, self.size = indices, size

    def run(self):
        caps = []
        try:
            for index in self.indices:
                cap = cv2.VideoCapture(index)
                caps.append(cap)
                if not cap.isOpened():
                    raise RuntimeError(
                        f"Cannot open camera {index}. Check index, OS permissions and other apps."
                    )
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Best effort; verify on your hardware.
            while not self.isInterruptionRequested():
                stamps = []
                for cap in caps:
                    if not cap.grab():
                        raise RuntimeError("Camera disconnected or frame grab failed")
                    stamps.append(monotonic())
                images = []
                for cap in caps:
                    ok, img = cap.retrieve()
                    if not ok or img is None:
                        raise RuntimeError("Frame retrieval failed")
                    images.append(img)
                if any((im.shape[1], im.shape[0]) != self.size for im in images):
                    raise RuntimeError(
                        "Camera did not supply requested resolution; select its actual supported size"
                    )
                self.frames.emit((images, stamps))
                self.msleep(50)
        except Exception as exc:
            self.failure.emit(str(exc))
        finally:
            for cap in caps:
                cap.release()
