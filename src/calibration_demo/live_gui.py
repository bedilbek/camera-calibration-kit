"""Two live previews and an orbitable 3D segment, using the existing Qt dependency."""

import sys
from time import monotonic

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .live import LiveWorker, load_live_rig


class SegmentView(QWidget):
    """Small orthographic 3D viewer; repaint only on new data or mouse interaction."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 300)
        self.result = None
        self.ids = (0, 1)
        self.center = None
        self.radius_m = 0.25
        self.yaw, self.pitch = -0.4, -0.35
        self.drag = None

    def set_result(self, result, ids):
        self.result, self.ids = result, ids
        if result and self.center is None:
            self.fit()
        self.update()

    def fit(self):
        if self.result:
            points = np.asarray(self.result["points_left_frame_m"])
            self.center = points.mean(axis=0)
            self.radius_m = max(float(np.linalg.norm(points[1] - points[0])) * 0.8, 0.05)
        self.update()

    def screen(self, point):
        cy, sy, cp, sp = np.cos(self.yaw), np.sin(self.yaw), np.cos(self.pitch), np.sin(self.pitch)
        rotation = np.array([[cy, 0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]])
        xyz = rotation @ (np.asarray(point) - self.center)
        scale = min(self.width(), self.height()) * 0.36 / self.radius_m
        return QPointF(self.width() / 2 + xyz[0] * scale, self.height() / 2 + xyz[1] * scale)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#101b2b"))
        p.setPen(QColor("#b1c5db"))
        p.drawText(18, 26, "3D • left camera coordinates")
        p.drawText(18, self.height() - 18, "Drag to orbit · Wheel to zoom · Fit view to recenter")
        if not self.result:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for a valid marker pair")
            return
        for i, (label, color) in enumerate(zip(("X", "Y", "Z"), ("#ef798a", "#79d6ad", "#72adff"))):
            endpoint = self.center.copy()
            endpoint[i] += self.radius_m * 0.65
            p.setPen(QPen(QColor(color), 1.5))
            p.drawLine(self.screen(self.center), self.screen(endpoint))
            p.drawText(self.screen(endpoint) + QPointF(5, -5), label)
        points = np.asarray(self.result["points_left_frame_m"])
        a, b = [self.screen(point) for point in points]
        p.setPen(QPen(QColor("#f5d474"), 3))
        p.drawLine(a, b)
        for marker_id, pos, color in zip(self.ids, (a, b), ("#6ae4c5", "#7eb9ff")):
            p.setBrush(QColor(color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(pos, 8, 8)
            p.setPen(QColor("#eef5ff"))
            p.drawText(pos + QPointF(12, -12), f"ID {marker_id}")
        midpoint = (a + b) / 2
        label = f"{self.result['length_m'] * 100:.0f} cm"
        box = QRectF(midpoint.x() - 75, midpoint.y() - 34, 150, 26)
        p.fillRect(box, QColor("#25384d"))
        p.setPen(QColor("#ffffff"))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, label)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag = event.position()

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            delta = event.position() - self.drag
            self.yaw += delta.x() * 0.01
            self.pitch = float(np.clip(self.pitch + delta.y() * 0.01, -1.5, 1.5))
            self.drag = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self.drag = None

    def wheelEvent(self, event):
        self.radius_m = float(
            np.clip(self.radius_m * 0.85 ** (event.angleDelta().y() / 120), 0.005, 100)
        )
        self.update()


class LiveWindow(QMainWindow):
    def __init__(self, pose=None, left_camera=0, right_camera=1):
        super().__init__()
        self.setWindowTitle("Live stereo marker distance")
        self.resize(1280, 860)
        self.worker = None
        self.stopping = False
        self.last_capture = 0
        self.rig = None
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        form = QFormLayout()
        layout.addLayout(form)
        self.inputs = {}
        self.config = []
        for key, label in (
            ("pose", "Sequential calibration.json"),
            ("left", "Left intrinsics (optional verification)"),
            ("right", "Right intrinsics (optional verification)"),
        ):
            row = QHBoxLayout()
            field = QLineEdit(str(pose) if key == "pose" and pose else "")
            field.setPlaceholderText("Select JSON; rig export already includes both intrinsics")
            self.inputs[key] = field
            row.addWidget(field)
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda checked=False, k=key: self.browse(k))
            row.addWidget(browse)
            form.addRow(label, row)
            self.config.extend((field, browse))
        row = QHBoxLayout()
        self.cameras, self.marker_ids = [], []
        for label, value, group, maximum in (
            ("Left camera index", left_camera, self.cameras, 100),
            ("Right camera index", right_camera, self.cameras, 100),
            ("Marker A ID", 0, self.marker_ids, 586),
            ("Marker B ID", 1, self.marker_ids, 586),
        ):
            row.addWidget(QLabel(label))
            field = QSpinBox()
            field.setRange(0, maximum)
            field.setValue(value)
            row.addWidget(field)
            group.append(field)
            self.config.append(field)
        layout.addLayout(row)
        self.confirm = QCheckBox(
            "Previews match left/right calibration; camera positions and lens settings are unchanged"
        )
        self.confirm.toggled.connect(self.clear_measurement)
        layout.addWidget(self.confirm)
        actions = QHBoxLayout()
        self.start_button = QPushButton("Start previews")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        fit = QPushButton("Fit 3D view")
        fit.clicked.connect(lambda: self.view.fit())
        for button in (self.start_button, self.stop_button, fit):
            actions.addWidget(button)
        self.readout = QLabel("Distance —")
        self.readout.setStyleSheet("font-size: 24px; font-weight: bold")
        actions.addWidget(self.readout, 1)
        layout.addLayout(actions)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        previews = QWidget()
        previews_layout = QVBoxLayout(previews)
        self.previews = []
        for title in ("LEFT / camera 1", "RIGHT / camera 2"):
            previews_layout.addWidget(QLabel(title))
            label = QLabel("Camera stopped")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(320, 180)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            label.setStyleSheet("background: #162436; color: #b1c5db")
            previews_layout.addWidget(label, 1)
            self.previews.append(label)
        splitter.addWidget(previews)
        self.view = SegmentView()
        splitter.addWidget(self.view)
        splitter.setSizes([540, 700])
        layout.addWidget(splitter, 1)
        self.status = QLabel("Load a sequential rig, select camera indices and two AprilTag IDs.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        note = QLabel(
            "AprilTag 36h11 · 2 border bits · Center-to-center distance. Software pairing is not exposure synchronization; move slowly. Verify against a known rigid length."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    def browse(self, key):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select calibration", "sessions", "JSON (*.json)"
        )
        if path:
            self.inputs[key].setText(path)

    def clear_measurement(self, *_):
        self.readout.setText("Distance —")
        self.view.set_result(None, tuple(field.value() for field in self.marker_ids))

    def start(self):
        if self.worker is not None:
            return
        try:
            cameras = tuple(field.value() for field in self.cameras)
            ids = tuple(field.value() for field in self.marker_ids)
            if cameras[0] == cameras[1] or ids[0] == ids[1]:
                raise ValueError("Choose two different cameras and two different marker IDs")
            rig = load_live_rig(
                *(self.inputs[key].text().strip() for key in ("pose", "left", "right"))
            )
        except (ValueError, OSError, KeyError, TypeError, cv2.error) as exc:
            self.status.setText(str(exc))
            return
        self.rig = rig
        self.confirm.setChecked(False)
        self.view.center = None
        self.last_capture = monotonic()
        self.stopping = False
        self.worker = LiveWorker(cameras, rig, ids)
        for field in self.config:
            field.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText(f"Opening cameras at {rig.size[0]} × {rig.size[1]}…")
        self.worker.start()

    def poll(self):
        if self.worker is None:
            return
        if self.stopping:
            if not self.worker.isRunning():
                self.stop()
            return
        packet = self.worker.take()
        if packet is not None:
            if "error" in packet:
                self.status.setText(packet["error"])
                self.stop(keep_status=True)
                return
            self.show_packet(packet)
        if monotonic() - self.last_capture > 0.75:
            self.clear_measurement()
            self.status.setText("Stream stalled / frames too old — measurement cleared")
        if self.worker is not None and not self.worker.isRunning():
            self.stop(keep_status=True)

    def show_packet(self, packet):
        self.last_capture = packet["captured_at"]
        result = packet["result"]
        fresh = monotonic() - self.last_capture <= 0.75
        valid = result is not None and fresh and self.confirm.isChecked()
        ids = tuple(field.value() for field in self.marker_ids)
        if valid:
            self.readout.setText(f"{result['length_m'] * 100:.0f} cm")
            self.view.set_result(result, ids)
        else:
            self.clear_measurement()
        for index, (label, image) in enumerate(zip(self.previews, packet["images"])):
            # Resize the preview before drawing to avoid full-resolution overlay copies.
            scale = min(
                max(label.width(), 1) / image.shape[1], max(label.height(), 1) / image.shape[0]
            )
            preview = cv2.resize(image, None, fx=scale, fy=scale)
            centers = packet["centers"][index] if index < len(packet["centers"]) else {}
            pixels = []
            for marker_id in ids:
                if marker_id in centers:
                    xy = tuple(np.rint(np.asarray(centers[marker_id]) * scale).astype(int))
                    pixels.append(xy)
                    cv2.circle(preview, xy, 5, (100, 240, 220), 2)
                    cv2.putText(
                        preview,
                        str(marker_id),
                        (xy[0] + 7, xy[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
            if len(pixels) == 2:
                cv2.line(preview, *pixels, (100, 210, 255), 2)
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            qimage = QImage(
                rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
            )
            label.setPixmap(QPixmap.fromImage(qimage))
        state = (
            packet["status"]
            if self.confirm.isChecked()
            else "Verify camera order and fixed pose using the checkbox"
        )
        if not fresh:
            state = "Frames too old — measurement cleared"
        diagnostic = ""
        if valid:
            diagnostic = f" · max residual {np.max(result['reprojection_px']):.2f} px · min ray {min(result['ray_angles_deg']):.1f}°"
        self.status.setText(
            f"{state} · processing {packet['fps']:.1f} pairs/s · host grab gap {packet['host_gap_ms']:.1f} ms (not exposure skew){diagnostic}"
        )

    def stop(self, checked=False, *, keep_status=False):
        self.stopping = True
        self.clear_measurement()
        self.confirm.setChecked(False)
        if self.worker is not None:
            self.worker.requestInterruption()
            if not self.worker.wait(1500):
                self.clear_measurement()
                self.status.setText(
                    "Waiting for camera driver to return; devices are still stopping…"
                )
                return False
            self.worker = None
        self.clear_measurement()
        self.confirm.setChecked(False)
        for field in self.config:
            field.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        for label in self.previews:
            label.clear()
            label.setText("Camera stopped")
        if not keep_status:
            self.status.setText("Stopped — cameras released")
        return True

    def closeEvent(self, event):
        if self.stop():
            event.accept()
        else:
            event.ignore()


def launch(pose=None, left_camera=0, right_camera=1):
    app = QApplication.instance() or QApplication(sys.argv)
    window = LiveWindow(pose, left_camera, right_camera)
    window.show()
    return app.exec()
