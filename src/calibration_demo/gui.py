"""Native camera capture and source-pixel annotation UI."""

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .calibration import Board, detect, intrinsic, stereo
from .camera_identity import normalize_usb_id, suggested_usb_camera, usb_devices
from .capture import CaptureWorker
from .geometry import Rig, measure


class ImagePane(QGraphicsView):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.image = None
        self.points = []
        self.annotating = False
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(320, 250)

    def show_image(self, image, fit=False):
        self.image = image.copy()
        self.scene().clear()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        q = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.scene().addPixmap(QPixmap.fromImage(q))
        self.setSceneRect(0, 0, w, h)
        for i, (x, y) in enumerate(self.points):
            color = QColor("#00b8ff" if i == 0 else "#ffb000")
            self.scene().addEllipse(x - 5, y - 5, 10, 10, QPen(color, 2))
            t = self.scene().addText("AB"[i])
            t.setDefaultTextColor(color)
            t.setPos(x + 7, y - 20)
        if len(self.points) == 2:
            self.scene().addLine(*self.points[0], *self.points[1], QPen(QColor("#00b8ff"), 2))
        if fit:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        if 0.05 < self.transform().m11() * factor < 30:
            self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        if (
            self.annotating
            and event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            p = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(p) and len(self.points) < 2:
                self.points.append((p.x(), p.y()))
                self.show_image(self.image)
                self.changed.emit()
            event.accept()
        elif self.annotating and event.button() == Qt.MouseButton.RightButton:
            if self.points:
                self.points.pop()
                self.show_image(self.image)
                self.changed.emit()
            event.accept()
        else:
            super().mousePressEvent(event)


class Window(QMainWindow):
    def __init__(self, session_root=None):
        super().__init__()
        self.setWindowTitle("Calibration Lab | camera calibration and 3D measurement")
        self.resize(1500, 920)
        self.worker = None
        self.latest = None
        self.frozen = False
        self.rig = None
        self.intrinsics = [None, None]
        self.observations = [[], []]
        self.pairs = []
        self.board = Board()
        self.report = None
        self.review_windows = []
        self.capture_ids = None
        self.verified_capture_ids = set()
        self.root = Path(
            session_root or ("sessions/" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
        )
        self.root.mkdir(parents=True, exist_ok=True)
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.addWidget(
            QLabel(
                "CALIBRATION LAB   •   Static scenes only   •   Metres   •   Left camera = reference frame"
            )
        )
        row = QHBoxLayout()
        self.single_camera = QCheckBox("Single camera (intrinsics)")
        row.addWidget(self.single_camera)
        self.ids = [QSpinBox(), QSpinBox()]
        self.ids[1].setValue(1)
        self.camera_labels = []
        for label, box in zip(("Left camera", "Right camera"), self.ids):
            box.setRange(0, 30)
            camera_label = QLabel(label)
            self.camera_labels.append(camera_label)
            row.addWidget(camera_label)
            row.addWidget(box)
        self.width, self.height = QSpinBox(), QSpinBox()
        for label, box, value in [("Width", self.width, 1280), ("Height", self.height, 720)]:
            box.setRange(100, 8000)
            box.setValue(value)
            row.addWidget(QLabel(label))
            row.addWidget(box)
        for title, fn in [
            ("Start cameras", self.start),
            ("Stop", self.stop),
            ("Freeze / resume", self.freeze),
            ("Open session", self.open_session),
        ]:
            button = self.button(row, title, fn)
            if title == "Start cameras":
                self.start_button = button
        layout.addLayout(row)
        identity_row = QHBoxLayout()
        self.usb_ids = [QLineEdit(), QLineEdit()]
        self.usb_labels = [QLabel("Camera USB ID"), QLabel("Right USB ID")]
        self.usb_confirms = [QCheckBox("Preview matches ID"), QCheckBox("Preview matches ID")]
        for label, field, confirm in zip(self.usb_labels, self.usb_ids, self.usb_confirms):
            field.setPlaceholderText("usb:VID:PID:SERIAL (run: uv run python run.py camera-id)")
            field.textChanged.connect(lambda _text, box=confirm: box.setChecked(False))
            confirm.setEnabled(False)
            identity_row.addWidget(label)
            identity_row.addWidget(field)
            identity_row.addWidget(confirm)
        layout.addLayout(identity_row)
        row = QHBoxLayout()
        self.squares_x, self.squares_y = QSpinBox(), QSpinBox()
        for label, box, value in (
            ("Board columns", self.squares_x, 7),
            ("Rows", self.squares_y, 5),
        ):
            box.setRange(3, 30)
            box.setValue(value)
            row.addWidget(QLabel(label))
            row.addWidget(box)
        self.square = QDoubleSpinBox()
        self.square.setRange(5, 200)
        self.square.setValue(40)
        row.addWidget(QLabel("Square (mm)"))
        row.addWidget(self.square)
        self.marker = QDoubleSpinBox()
        self.marker.setRange(1, 199)
        self.marker.setValue(28)
        row.addWidget(QLabel("Marker (mm)"))
        row.addWidget(self.marker)
        self.dictionary = QComboBox()
        self.dictionary.addItems(
            [
                "DICT_4X4_50",
                "DICT_4X4_100",
                "DICT_4X4_250",
                "DICT_4X4_1000",
                "DICT_5X5_100",
                "DICT_5X5_250",
                "DICT_6X6_250",
                "DICT_7X7_250",
            ]
        )
        self.dictionary.setCurrentText("DICT_4X4_100")
        row.addWidget(QLabel("Dictionary"))
        row.addWidget(self.dictionary)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.stereo_controls = []
        for title, fn in [
            ("Save board PNG", self.save_board),
            ("Capture left board", lambda: self.capture_board(0)),
            ("Capture right board", lambda: self.capture_board(1)),
            ("Solve intrinsics", self.solve_intrinsics),
            ("Capture stereo board", self.capture_pair),
            ("Solve stereo", self.solve_stereo),
        ]:
            button = self.button(row, title, fn)
            if title == "Capture left board":
                self.capture_left_button = button
            if title in ("Capture right board", "Capture stereo board", "Solve stereo"):
                self.stereo_controls.append(button)
        layout.addLayout(row)
        row = QHBoxLayout()
        for title, fn in [
            ("Load calibration", self.load_rig),
            ("Save calibration", self.save_rig),
            ("Load intrinsics", self.load_intrinsics),
            ("Review intrinsics", self.open_review),
            ("Reset samples", self.reset_samples),
            ("Clear points", self.clear_points),
            ("Fit images", self.fit_images),
        ]:
            self.button(row, title, fn)
        layout.addLayout(row)
        self.workflow_hint = QLabel(
            "Freeze first. Ctrl+click A then B in EACH image. Wheel: zoom. Drag: pan. Right click: undo."
        )
        layout.addWidget(self.workflow_hint)
        row = QHBoxLayout()
        self.panes = [ImagePane(), ImagePane()]
        self.pane_containers = []
        self.pane_labels = []
        for name, pane in zip(("LEFT • A then B", "RIGHT • same A then same B"), self.panes):
            container = QWidget()
            col = QVBoxLayout(container)
            label = QLabel(name)
            self.pane_containers.append(container)
            self.pane_labels.append(label)
            col.addWidget(label)
            col.addWidget(pane)
            row.addWidget(container)
            pane.changed.connect(self.invalidate_report)
        layout.addLayout(row, 1)
        self.measurement_controls = QWidget()
        row = QHBoxLayout(self.measurement_controls)
        self.reference = QDoubleSpinBox()
        self.reference.setRange(0, 1000)
        self.reference.setDecimals(4)
        self.reference.setSpecialValueText("No reference")
        self.reference.valueChanged.connect(self.invalidate_report)
        row.addWidget(QLabel("Tape measurement (m)"))
        row.addWidget(self.reference)
        self.button(row, "Measure A–B", self.calculate)
        self.button(row, "Export session + result", self.export_session)
        layout.addWidget(self.measurement_controls)
        self.readout = QLabel("Freeze a calibrated pair to measure A–B")
        self.readout.setStyleSheet(
            "font-size: 27px; font-weight: bold; color: #187BC1; padding: 8px;"
        )
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(190)
        layout.addWidget(self.log)
        self.setCentralWidget(host)
        self.single_camera.toggled.connect(self.change_capture_mode)
        self.say(
            "Use the synthetic session to rehearse without cameras. Calibrate at the exact working resolution."
        )
        self.setStyleSheet("QWidget {font-size: 13px;} QPushButton {padding: 6px;}")

    def button(self, row, text, fn):
        b = QPushButton(text)
        b.clicked.connect(lambda: self.guard(fn))
        row.addWidget(b)
        return b

    def change_capture_mode(self):
        single = self.single_camera.isChecked()
        self.camera_labels[0].setText("Camera" if single else "Left camera")
        self.camera_labels[1].setVisible(not single)
        self.usb_labels[1].setVisible(not single)
        self.usb_ids[1].setVisible(not single)
        self.usb_confirms[1].setVisible(not single)
        self.ids[1].setEnabled(not single)
        self.ids[1].setVisible(not single)
        self.start_button.setText("Start camera" if single else "Start cameras")
        self.capture_left_button.setText("Capture board photo" if single else "Capture left board")
        self.pane_containers[1].setVisible(not single)
        self.measurement_controls.setVisible(not single)
        self.pane_labels[0].setText("CAMERA • calibration board" if single else "LEFT • A then B")
        self.workflow_hint.setText(
            "Start camera → verify preview and USB ID → capture 20–30 board views → solve intrinsics."
            if single
            else "Freeze first. Ctrl+click A then B in EACH image. Wheel: zoom. Drag: pan. Right click: undo."
        )
        for control in self.stereo_controls:
            control.setEnabled(not single)
            control.setVisible(not single)
        self.latest = None
        self.frozen = False
        self.reset_samples()
        if single and not self.usb_ids[0].text().strip():
            try:
                suggested = suggested_usb_camera(usb_devices())
            except Exception:
                suggested = None
            if suggested:
                self.usb_ids[0].setText(suggested)
                self.say("USB ID suggested. Check the live preview before you confirm it.")
        for pane in self.panes:
            pane.points = []
            pane.image = None
            pane.scene().clear()
        self.invalidate_report()
        self.say("Capture mode changed. Collect new observations for the selected camera(s).")

    def guard(self, fn):
        try:
            fn()
        except Exception as exc:
            self.say(f"ERROR: {exc}")
            QMessageBox.warning(self, "Check this step", str(exc))

    def say(self, text):
        self.log.appendPlainText(text)

    def invalidate_report(self, *_):
        self.report = None
        self.readout.setText(
            "Collect 20–30 diverse board views, then solve intrinsics"
            if self.single_camera.isChecked()
            else "Select matching A and B, then measure"
        )

    def image_size(self):
        return self.width.value(), self.height.value()

    def current_board(self):
        b = Board(
            squares_x=self.squares_x.value(),
            squares_y=self.squares_y.value(),
            square_m=self.square.value() / 1000,
            marker_m=self.marker.value() / 1000,
            dictionary=self.dictionary.currentText(),
        )
        if b.marker_m >= b.square_m:
            raise ValueError("Marker size must be smaller than board square size")
        if b != self.board:
            if any(self.observations) or self.pairs or any(x is not None for x in self.intrinsics):
                raise ValueError("Reset samples before changing board dimensions")
            self.board = b
        return b

    def start(self):
        if self.worker is not None:
            raise ValueError("Stop existing cameras first")
        ids = [x.value() for x in self.ids[: 1 if self.single_camera.isChecked() else 2]]
        identities = [
            normalize_usb_id(field.text()) if field.text().strip() else None
            for field in self.usb_ids[: len(ids)]
        ]
        present = {device["id"] for device in usb_devices()} if any(identities) else set()
        if any(identity not in present for identity in identities if identity):
            raise ValueError("The entered USB camera ID is not connected. Run camera-id again.")
        if len(identities) == 2 and identities[0] and identities[0] == identities[1]:
            raise ValueError("The two camera IDs must differ")
        if any(self.observations) and self.capture_ids != identities:
            raise ValueError("Camera ID changed while samples exist. Reset samples first.")
        if len(ids) == 2 and ids[0] == ids[1]:
            raise ValueError("Choose two different camera indices")
        if self.rig and self.rig.metadata.get("synthetic"):
            raise ValueError(
                "Reset samples and load or solve a REAL calibration before starting cameras"
            )
        if self.rig and self.image_size() != self.rig.size:
            raise ValueError("Resolution does not match calibration")
        self.latest = None
        self.frozen = False
        self.clear_points()
        self.worker = CaptureWorker(ids, self.image_size())
        self.capture_ids = identities
        for confirm in self.usb_confirms:
            confirm.setChecked(False)
            confirm.setEnabled(False)
        self.worker.frames.connect(self.on_frames)
        self.worker.failure.connect(self.on_failure)
        self.worker.start()
        for control in [self.single_camera, *self.ids, *self.usb_ids, self.width, self.height]:
            control.setEnabled(False)
        self.say(
            "Capture started. Verify autofocus is locked and camera order matches calibration."
        )
        if any(identity is None for identity in identities):
            self.say(
                "Camera USB ID is missing. This session cannot prove physical camera identity."
            )
        else:
            self.say("When the preview appears, check Preview matches ID before board capture.")

    def stop(self):
        if self.worker:
            self.worker.requestInterruption()
            if not self.worker.wait(2000):
                self.say("Camera backend is still stopping; wait before restarting.")
                return False
            self.worker = None
        for control in [self.single_camera, *self.ids, *self.usb_ids, self.width, self.height]:
            control.setEnabled(True)
        self.ids[1].setEnabled(not self.single_camera.isChecked())
        return True

    def on_failure(self, text):
        self.latest = None
        self.frozen = False
        self.clear_points()
        self.say("CAMERA ERROR: " + text)

    def on_frames(self, packet):
        if self.frozen or self.worker is None:
            return
        first = self.latest is None
        self.latest = packet
        if first:
            for index, confirm in enumerate(self.usb_confirms[: len(packet[0])]):
                confirm.setEnabled(bool(self.capture_ids and self.capture_ids[index]))
        for pane, img in zip(self.panes, packet[0]):
            pane.annotating = False
            pane.show_image(img, fit=first)

    def freeze(self):
        if self.latest is None:
            raise ValueError("No frames available")
        self.frozen = not self.frozen
        self.clear_points()
        for pane in self.panes:
            pane.annotating = self.frozen
        self.say("Frozen: annotate matching endpoints." if self.frozen else "Live: points cleared.")

    def require_frames(self):
        if self.latest is None:
            raise ValueError("Start cameras or open a session")
        return self.latest[0]

    def capture_board(self, index):
        images = self.require_frames()
        if index >= len(images):
            raise ValueError("Only one camera is active; use Capture board photo")
        if self.capture_ids and self.capture_ids[index]:
            if not self.usb_confirms[index].isChecked():
                raise ValueError("Check the live preview, then select Preview matches ID")
        board = self.current_board()
        if self.frozen:
            raise ValueError("Resume before collecting a new pose")
        obs = detect(images[index], board)
        folder = self.root / f"intrinsic_{index}"
        folder.mkdir(exist_ok=True)
        image_path = folder / f"{len(self.observations[index]) + 1:03d}.png"
        if image_path.exists():
            raise FileExistsError(
                f"Camera image already exists: {image_path}. Start a new session."
            )
        if not cv2.imwrite(str(image_path), images[index]):
            raise OSError(f"Cannot save camera image: {image_path}")
        self.observations[index].append(obs)
        if self.capture_ids and self.capture_ids[index]:
            self.verified_capture_ids.add(index)
        self.say(
            f"Camera {index}: {len(self.observations[index])} views; {len(obs[0])} corners. Vary tilt, depth and coverage."
        )
        if self.single_camera.isChecked():
            self.readout.setText(
                f"{len(self.observations[index])} board views captured — aim for 20–30"
            )

    def solve_intrinsics(self):
        self.current_board()
        count = 1 if self.single_camera.isChecked() else 2
        solved = [intrinsic(o, self.image_size(), self.board) for o in self.observations[:count]]
        self.intrinsics = solved + [None] * (2 - count)
        self.rig = None
        self.invalidate_report()
        payload = [dict(K=k.tolist(), d=d.tolist(), metadata=meta) for k, d, meta in solved]
        (self.root / "intrinsics.json").write_text(
            json.dumps(
                dict(
                    board=vars(self.board),
                    cameras=payload,
                    camera_indices=[x.value() for x in self.ids[:count]],
                    camera_identities=self.capture_ids or [None] * count,
                    camera_identity_source=(
                        "preview_confirmed_at_capture"
                        if self.capture_ids
                        and all(self.capture_ids)
                        and all(index in self.verified_capture_ids for index in range(count))
                        else "missing"
                    ),
                ),
                indent=2,
            )
        )
        self.say("Intrinsics saved: " + str(self.root / "intrinsics.json"))
        self.say(json.dumps([x[2] for x in solved], indent=2))
        if count == 1:
            self.readout.setText(
                f"Intrinsics saved | {solved[0][2]['views']} views | RMS {solved[0][2]['rms_px']:.3f} px"
            )

    def load_intrinsics(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load intrinsics", "", "JSON (*.json)")
        if path:
            obj = json.loads(Path(path).read_text())
            values = [
                (np.asarray(c["K"]), np.asarray(c["d"]), c["metadata"]) for c in obj["cameras"]
            ]
            count = 1 if self.single_camera.isChecked() else 2
            if len(values) != count or any(
                tuple(v[2]["size"]) != self.image_size() for v in values
            ):
                raise ValueError(f"Need {count} calibration(s) at the selected resolution")
            self.board = Board(**obj["board"])
            self.squares_x.setValue(self.board.squares_x)
            self.squares_y.setValue(self.board.squares_y)
            self.square.setValue(self.board.square_m * 1000)
            self.marker.setValue(self.board.marker_m * 1000)
            if self.dictionary.findText(self.board.dictionary) < 0:
                self.dictionary.addItem(self.board.dictionary)
            self.dictionary.setCurrentText(self.board.dictionary)
            self.intrinsics = values + [None] * (2 - count)
            self.pairs = []
            self.rig = None
            self.invalidate_report()
            self.say(
                "Single-camera intrinsics loaded."
                if count == 1
                else "Intrinsics loaded. Capture new stereo poses with the cameras fixed."
            )

    def open_review(self):
        from .review_gui import ReviewWindow

        session = self.root
        if not (session / "intrinsics.json").is_file():
            chosen = QFileDialog.getExistingDirectory(
                self, "Choose a saved intrinsic session", "sessions"
            )
            if not chosen:
                return
            session = Path(chosen)
        review = ReviewWindow(session)
        self.review_windows.append(review)
        review.show()

    def capture_pair(self):
        if self.frozen:
            raise ValueError("Resume and hold the board still for each capture")
        images = self.require_frames()
        if len(images) != 2:
            raise ValueError("Stereo calibration requires two cameras")
        board = self.current_board()
        pair = [detect(im, board) for im in images]
        if len(np.intersect1d(pair[0][0], pair[1][0])) < 6:
            raise ValueError("Need six common corner IDs in both views")
        self.pairs.append(pair)
        folder = self.root / "stereo"
        folder.mkdir(exist_ok=True)
        for i, im in enumerate(images):
            cv2.imwrite(str(folder / f"{len(self.pairs):03d}_{i}.png"), im)
        self.say(
            f"{len(self.pairs)} stereo poses. Board must be stationary during software-paired capture."
        )

    def solve_stereo(self):
        if any(x is None for x in self.intrinsics):
            raise ValueError("Solve or load intrinsics first")
        self.current_board()
        self.rig = stereo(self.pairs, self.image_size(), self.board, *self.intrinsics)
        self.rig.metadata["camera_indices"] = [x.value() for x in self.ids]
        self.rig.save(self.root / "calibration.json")
        self.invalidate_report()
        self.say(
            f"Stereo RMS {self.rig.metadata['stereo_rms_px']:.3f} px | baseline {np.linalg.norm(self.rig.T):.4f} m. Validate with an independent known length."
        )

    def reset_samples(self):
        self.capture_ids = None
        self.verified_capture_ids.clear()
        self.observations = [[], []]
        self.pairs = []
        self.intrinsics = [None, None]
        self.rig = None
        self.report = None
        self.board = Board(
            squares_x=self.squares_x.value(),
            squares_y=self.squares_y.value(),
            square_m=self.square.value() / 1000,
            marker_m=self.marker.value() / 1000,
            dictionary=self.dictionary.currentText(),
        )
        self.say("Samples and calibration reset. Saved files remain.")

    def save_board(self):
        b = self.current_board()
        path, _ = QFileDialog.getSaveFileName(self, "Save board", "charuco.png", "PNG (*.png)")
        if path:
            b.write(path)
            self.say(
                f"Print board area {b.square_m * b.squares_x * 1000:g} × "
                f"{b.square_m * b.squares_y * 1000:g} mm. "
                f"Verify square {b.square_m * 1000:g} mm. Mount flat."
            )

    def load_rig(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load calibration", "", "JSON (*.json)")
        if path:
            rig = Rig.load(path)
            if self.latest and any((im.shape[1], im.shape[0]) != rig.size for im in self.latest[0]):
                raise ValueError("Image resolution differs from calibration")
            if self.worker and rig.metadata.get("synthetic"):
                raise ValueError("Cannot apply synthetic calibration to live cameras")
            self.rig = rig
            self.width.setValue(rig.size[0])
            self.height.setValue(rig.size[1])
            self.invalidate_report()
            self.say(
                "Calibration loaded; confirm physical camera order and unchanged lens settings."
            )

    def save_rig(self):
        if not self.rig:
            raise ValueError("No calibration")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save calibration", "calibration.json", "JSON (*.json)"
        )
        if path:
            self.rig.save(path)

    def clear_points(self):
        for pane in self.panes:
            pane.points = []
            if pane.image is not None:
                pane.show_image(pane.image)
        self.invalidate_report()

    def fit_images(self):
        for pane in self.panes:
            pane.fitInView(pane.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def calculate(self):
        if not self.frozen or not self.rig:
            raise ValueError("Freeze images and load or solve calibration first")
        if any((im.shape[1], im.shape[0]) != self.rig.size for im in self.require_frames()):
            raise ValueError("Images do not match calibrated resolution")
        self.report = measure(
            self.rig, self.panes[0].points, self.panes[1].points, self.reference.value() or None
        )
        tag = "SYNTHETIC  |  " if self.rig.metadata.get("synthetic") else ""
        summary = f"{tag}A–B: {self.report['length_m']:.4f} m"
        if "absolute_error_m" in self.report:
            summary += f"  |  tape error: {1000 * self.report['absolute_error_m']:.1f} mm"
        summary += f"  |  max residual: {np.max(self.report['reprojection_px']):.2f} px"
        if self.report["warnings"]:
            summary += "  |  CHECK GEOMETRY"
        self.readout.setText(summary)
        self.say(json.dumps(self.report, indent=2))
        if self.rig.metadata.get("synthetic"):
            self.say("SYNTHETIC REHEARSAL: not a real-world accuracy result.")

    def export_session(self):
        if not self.frozen or not self.rig:
            raise ValueError("Freeze a calibrated image pair first")
        directory = QFileDialog.getExistingDirectory(self, "Choose parent for new session folder")
        if not directory:
            return
        out = Path(directory) / datetime.now(timezone.utc).strftime("measurement-%Y%m%d-%H%M%S-%f")
        out.mkdir()
        for name, im in zip(("left.png", "right.png"), self.require_frames()):
            if not cv2.imwrite(str(out / name), im):
                raise OSError("Image save failed")
        self.rig.save(out / "calibration.json")
        payload = dict(
            schema_version=1,
            saved_at_utc=datetime.now(timezone.utc).isoformat(),
            points=[p.points for p in self.panes],
            reference_m=self.reference.value() or None,
            host_grab_times=self.latest[1],
            timing_note="Host timestamps are not exposure times. Static scenes only.",
            result=self.report,
        )
        (out / "session.json").write_text(json.dumps(payload, indent=2))
        self.say(f"Exported {out}")

    def open_session(self):
        path = QFileDialog.getExistingDirectory(self, "Choose folder containing session.json")
        if path:
            self.read_session(path)

    def read_session(self, directory):
        directory = Path(directory)
        rig = Rig.load(directory / "calibration.json")
        obj = json.loads((directory / "session.json").read_text())
        images = [cv2.imread(str(directory / n)) for n in ("left.png", "right.png")]
        if any(im is None or (im.shape[1], im.shape[0]) != rig.size for im in images):
            raise ValueError("Missing images or resolution mismatch")
        if not self.stop():
            raise ValueError("Wait for camera shutdown before loading a session")
        self.single_camera.setChecked(False)
        self.rig = rig
        self.latest = (images, obj.get("host_grab_times", []))
        self.frozen = True
        self.width.setValue(rig.size[0])
        self.height.setValue(rig.size[1])
        self.reference.setValue(obj.get("reference_m") or 0)
        for pane, im, points in zip(self.panes, images, obj.get("points", [[], []])):
            pane.points = points
            pane.annotating = True
            pane.show_image(im, fit=True)
        self.report = None
        self.say(
            "Session loaded. "
            + ("SYNTHETIC rehearsal." if rig.metadata.get("synthetic") else "Recorded pair.")
        )

    def closeEvent(self, event):
        if self.stop():
            event.accept()
        else:
            event.ignore()


def launch(session=None, screenshot=None, single_camera=False, camera=0):
    app = QApplication.instance() or QApplication([])
    win = Window()
    win.ids[0].setValue(camera)
    win.single_camera.setChecked(single_camera)
    win.show()
    if session:
        win.read_session(session)
    if screenshot:

        def capture():
            win.calculate()
            win.grab().save(str(screenshot))
            win.close()
            app.quit()

        QTimer.singleShot(700, capture)
    return app.exec()
