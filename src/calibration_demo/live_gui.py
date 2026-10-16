"""Two live previews and an orbitable 3D segment, using the existing Qt dependency."""

import sys
from time import monotonic

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
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
from .view_geometry import (
    camera_outline,
    ground_grid,
    ground_projection,
    look_at,
    observer_pose,
    rig_frame,
)


class SegmentView(QWidget):
    """Perspective observer above and behind the calibrated camera pair."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 300)
        self.result = None
        self.rig = None
        self.ids = (0, 1)
        self.center = None
        self.distance_m = 1.1
        self.yaw = self.pitch = 0.0
        self.fov_deg = 65.0
        self.drag = None

    def set_rig(self, rig):
        self.rig = rig
        self.result = None
        self.center = None
        self.yaw = self.pitch = 0.0
        self.fov_deg = 65.0
        self.update()

    def set_result(self, result, ids):
        self.result, self.ids = result, ids
        if result and self.center is None:
            self.fit()
        self.update()

    def set_distance(self, distance_m):
        self.distance_m = distance_m
        self.yaw = self.pitch = 0.0
        self.fit()

    def pose(self):
        eye, up, forward = observer_pose(
            self.rig, self.distance_m, self.distance_m, self.yaw, self.pitch
        )
        centers, _, _ = rig_frame(self.rig)
        target = self.center if self.center is not None else centers.mean(axis=0) + forward
        return eye, look_at(eye, target, up)

    def scene_points(self):
        points = np.vstack([camera_outline(self.rig, i) for i in (0, 1)])
        if self.result:
            points = np.vstack([points, self.result["points_left_frame_m"]])
        corners, _, _ = ground_grid(self.rig, points, self.distance_m)
        return np.vstack([points, corners])

    def fit(self):
        self.yaw = self.pitch = 0.0
        if self.rig is not None:
            points = self.scene_points()
            self.center = (points.min(axis=0) + points.max(axis=0)) / 2
            eye, rotation = self.pose()
            xyz = (rotation @ (self.scene_points() - eye).T).T
            visible = xyz[xyz[:, 2] > 0.01]
            if len(visible):
                aspect = max(self.width(), 1) / max(self.height(), 1)
                tangent = max(
                    np.max(np.abs(visible[:, 0]) / visible[:, 2]) / aspect,
                    np.max(np.abs(visible[:, 1]) / visible[:, 2]),
                )
                self.fov_deg = float(np.clip(np.degrees(2 * np.arctan(tangent * 1.4)), 25, 150))
        self.update()

    def screen(self, point):
        eye, rotation = self.pose()
        xyz = rotation @ (np.asarray(point) - eye)
        if xyz[2] <= 0.01:
            return None
        focal = self.height() / (2 * np.tan(np.radians(self.fov_deg) / 2))
        return QPointF(
            self.width() / 2 + xyz[0] * focal / xyz[2],
            self.height() / 2 - xyz[1] * focal / xyz[2],
        )

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#101b2b"))
        p.setPen(QColor("#b1c5db"))
        mode = "Observer preset" if self.yaw == self.pitch == 0 else "Orbit view"
        p.drawText(18, 26, f"3D · {mode} · level cameras assumed")
        p.drawText(
            18,
            49,
            f"Preset: {self.distance_m * 100:.0f} cm behind / above · left lens height {self.distance_m * 100:.0f} cm",
        )
        p.drawText(
            18, self.height() - 18, "Drag to orbit · Wheel to zoom · Fit 3D view restores observer"
        )
        if self.rig is None:
            p.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Start previews to load camera geometry"
            )
            return
        # Keep geometry and labels out of the view captions.
        p.setClipRect(QRectF(0, 60, self.width(), max(self.height() - 115, 1)))
        centers, _, _ = rig_frame(self.rig)
        self.draw_ground(p, centers)
        a, b = [self.screen(point) for point in centers]
        p.setPen(QPen(QColor("#6b8197"), 1, Qt.PenStyle.DashLine))
        if a is not None and b is not None:
            p.drawLine(a, b)
            p.drawText(
                (a + b) / 2 + QPointF(0, 20),
                f"Cameras: {np.linalg.norm(centers[1] - centers[0]) * 100:.1f} cm",
            )
        for camera, color, title in ((0, "#6ae4c5", "Left camera"), (1, "#7eb9ff", "Right camera")):
            points = [self.screen(point) for point in camera_outline(self.rig, camera)]
            p.setPen(QPen(QColor(color), 1.5))
            for i, j in ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)):
                if points[i] is not None and points[j] is not None:
                    p.drawLine(points[i], points[j])
            if points[0] is not None:
                p.setBrush(QColor(color))
                p.drawEllipse(points[0], 5, 5)
                p.drawText(points[0] + QPointF(8, -12), title)
        if not self.result:
            p.setPen(QColor("#b1c5db"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for a valid marker pair")
            return
        a, b = [self.screen(point) for point in self.result["points_left_frame_m"]]
        if a is None or b is None:
            return
        p.setPen(QPen(QColor("#f5d474"), 3))
        p.drawLine(a, b)
        for index, (marker_id, pos, color) in enumerate(
            zip(self.ids, (a, b), ("#6ae4c5", "#7eb9ff"))
        ):
            p.setBrush(QColor(color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(pos, 8, 8)
            p.setPen(QColor("#eef5ff"))
            text = f"ID {marker_id}"
            dx = -p.fontMetrics().horizontalAdvance(text) - 12 if index == 0 else 12
            p.drawText(pos + QPointF(dx, 20), text)
        midpoint = (a + b) / 2
        box = QRectF(midpoint.x() - 65, midpoint.y() - 34, 130, 26)
        p.fillRect(box, QColor("#25384d"))
        p.setPen(QColor("#ffffff"))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, f"{self.result['length_m'] * 100:.0f} cm")

    def draw_ground(self, painter, centers):
        markers = self.result["points_left_frame_m"] if self.result else []
        corners, lines, step_m = ground_grid(self.rig, markers, self.distance_m)
        projected = [self.screen(point) for point in corners]
        if all(point is not None for point in projected):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#1b3040"))
            painter.drawPolygon(QPolygonF(projected))
        painter.setPen(QPen(QColor("#355063"), 1))
        for start, end in lines:
            a, b = self.screen(start), self.screen(end)
            if a is not None and b is not None:
                painter.drawLine(a, b)
        # Dashed verticals connect lenses and markers to their ground footprints.
        for index, point in enumerate(np.vstack([centers, np.asarray(markers).reshape(-1, 3)])):
            foot, height = ground_projection(self.rig, point, self.distance_m)
            a, b = self.screen(point), self.screen(foot)
            if a is None or b is None:
                continue
            color = QColor("#6b8197" if index < 2 else "#d7c38b")
            painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
            painter.drawLine(a, b)
            painter.setBrush(color)
            painter.drawEllipse(b, 3, 3)
            if index >= 2:
                text = f"{height * 100:.0f} cm"
                dx = -painter.fontMetrics().horizontalAdvance(text) - 10 if index == 2 else 10
                painter.drawText((a + b) / 2 + QPointF(dx, 0), text)
        painter.setClipping(False)
        painter.setPen(QColor("#8da6b9"))
        painter.drawText(
            18,
            self.height() - 39,
            f"Assumed ground · grid {step_m * 100:g} cm · dashed verticals show height",
        )
        painter.setClipRect(QRectF(0, 60, self.width(), max(self.height() - 115, 1)))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag = event.position()

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            delta = event.position() - self.drag
            self.yaw += delta.x() * 0.01
            self.pitch = float(np.clip(self.pitch + delta.y() * 0.01, -1.0, 1.0))
            self.drag = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self.drag = None

    def wheelEvent(self, event):
        self.fov_deg = float(np.clip(self.fov_deg * 0.9 ** (event.angleDelta().y() / 120), 15, 150))
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
        observer_controls = QHBoxLayout()
        observer_controls.addWidget(QLabel("Observer offset / camera height"))
        self.observer_distance = QDoubleSpinBox()
        self.observer_distance.setRange(5, 500)
        self.observer_distance.setValue(110)
        self.observer_distance.setSuffix(" cm")
        self.observer_distance.setToolTip(
            "Sets observer distance behind and above the lens midpoint, and assumed left lens height above ground."
        )
        self.observer_distance.valueChanged.connect(self.update_observer)
        observer_controls.addWidget(self.observer_distance)
        observer_controls.addWidget(QLabel("Level cameras assumed · height anchored to left lens"))
        observer_controls.addStretch()
        layout.addLayout(observer_controls)
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

    def update_observer(self, *_):
        self.view.set_distance(self.observer_distance.value() / 100)

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
        self.view.set_rig(rig)
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
