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
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .calibration import Board, detect, intrinsic, stereo
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
        self.setWindowTitle("Calibration Lab | two cameras, one metric world")
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
        self.ids = [QSpinBox(), QSpinBox()]
        self.ids[1].setValue(1)
        for label, box in zip(("Left camera", "Right camera"), self.ids):
            box.setRange(0, 30)
            row.addWidget(QLabel(label))
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
            self.button(row, title, fn)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.square = QDoubleSpinBox()
        self.square.setRange(5, 200)
        self.square.setValue(40)
        row.addWidget(QLabel("7×5 board square (mm)"))
        row.addWidget(self.square)
        for title, fn in [
            ("Save board PNG", self.save_board),
            ("Capture left board", lambda: self.capture_board(0)),
            ("Capture right board", lambda: self.capture_board(1)),
            ("Solve intrinsics", self.solve_intrinsics),
            ("Capture stereo board", self.capture_pair),
            ("Solve stereo", self.solve_stereo),
        ]:
            self.button(row, title, fn)
        layout.addLayout(row)
        row = QHBoxLayout()
        for title, fn in [
            ("Load calibration", self.load_rig),
            ("Save calibration", self.save_rig),
            ("Load intrinsics", self.load_intrinsics),
            ("Reset samples", self.reset_samples),
            ("Clear points", self.clear_points),
            ("Fit images", self.fit_images),
        ]:
            self.button(row, title, fn)
        layout.addLayout(row)
        layout.addWidget(
            QLabel(
                "Freeze first. Ctrl+click A then B in EACH image. Wheel: zoom. Drag: pan. Right click: undo."
            )
        )
        row = QHBoxLayout()
        self.panes = [ImagePane(), ImagePane()]
        for name, pane in zip(("LEFT • A then B", "RIGHT • same A then same B"), self.panes):
            col = QVBoxLayout()
            col.addWidget(QLabel(name))
            col.addWidget(pane)
            row.addLayout(col)
            pane.changed.connect(self.invalidate_report)
        layout.addLayout(row, 1)
        row = QHBoxLayout()
        self.reference = QDoubleSpinBox()
        self.reference.setRange(0, 1000)
        self.reference.setDecimals(4)
        self.reference.setSpecialValueText("No reference")
        self.reference.valueChanged.connect(self.invalidate_report)
        row.addWidget(QLabel("Tape measurement (m)"))
        row.addWidget(self.reference)
        self.button(row, "Measure A–B", self.calculate)
        self.button(row, "Export session + result", self.export_session)
        layout.addLayout(row)
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
        self.say(
            "Use the synthetic session to rehearse without cameras. Calibrate at the exact working resolution."
        )
        self.setStyleSheet(
            "QWidget {font-size: 13px;} QPushButton {padding: 6px;} QMainWindow {background:#f4f6f8;}"
        )

    def button(self, row, text, fn):
        b = QPushButton(text)
        b.clicked.connect(lambda: self.guard(fn))
        row.addWidget(b)

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
        self.readout.setText("Select matching A and B, then measure")

    def image_size(self):
        return self.width.value(), self.height.value()

    def current_board(self):
        b = Board(square_m=self.square.value() / 1000, marker_m=self.square.value() * 0.7 / 1000)
        if b != self.board:
            if any(self.observations) or self.pairs or any(x is not None for x in self.intrinsics):
                raise ValueError("Reset samples before changing board size")
            self.board = b
        return b

    def start(self):
        if self.worker is not None:
            raise ValueError("Stop existing cameras first")
        ids = [x.value() for x in self.ids]
        if ids[0] == ids[1]:
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
        self.worker.frames.connect(self.on_frames)
        self.worker.failure.connect(self.on_failure)
        self.worker.start()
        for control in [*self.ids, self.width, self.height]:
            control.setEnabled(False)
        self.say(
            "Capture started. Verify autofocus is locked and camera order matches calibration."
        )

    def stop(self):
        if self.worker:
            self.worker.requestInterruption()
            if not self.worker.wait(2000):
                self.say("Camera backend is still stopping; wait before restarting.")
                return False
            self.worker = None
        for control in [*self.ids, self.width, self.height]:
            control.setEnabled(True)
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
        board = self.current_board()
        if self.frozen:
            raise ValueError("Resume before collecting a new pose")
        obs = detect(images[index], board)
        self.observations[index].append(obs)
        folder = self.root / f"intrinsic_{index}"
        folder.mkdir(exist_ok=True)
        cv2.imwrite(str(folder / f"{len(self.observations[index]):03d}.png"), images[index])
        self.say(
            f"Camera {index}: {len(self.observations[index])} views; {len(obs[0])} corners. Vary tilt, depth and coverage."
        )

    def solve_intrinsics(self):
        self.current_board()
        self.intrinsics = [intrinsic(o, self.image_size(), self.board) for o in self.observations]
        self.rig = None
        self.invalidate_report()
        payload = [
            dict(K=k.tolist(), d=d.tolist(), metadata=meta) for k, d, meta in self.intrinsics
        ]
        (self.root / "intrinsics.json").write_text(
            json.dumps(dict(board=vars(self.board), cameras=payload), indent=2)
        )
        self.say("Intrinsics saved: " + str(self.root / "intrinsics.json"))
        self.say(json.dumps([x[2] for x in self.intrinsics], indent=2))

    def load_intrinsics(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load intrinsics", "", "JSON (*.json)")
        if path:
            obj = json.loads(Path(path).read_text())
            values = [
                (np.asarray(c["K"]), np.asarray(c["d"]), c["metadata"]) for c in obj["cameras"]
            ]
            if len(values) != 2 or any(tuple(v[2]["size"]) != self.image_size() for v in values):
                raise ValueError("Need two calibrations at the selected resolution")
            self.board = Board(**obj["board"])
            self.square.setValue(self.board.square_m * 1000)
            self.intrinsics = values
            self.pairs = []
            self.rig = None
            self.invalidate_report()
            self.say("Intrinsics loaded. Capture new stereo poses with the cameras fixed.")

    def capture_pair(self):
        if self.frozen:
            raise ValueError("Resume and hold the board still for each capture")
        images = self.require_frames()
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
        self.observations = [[], []]
        self.pairs = []
        self.intrinsics = [None, None]
        self.rig = None
        self.report = None
        self.board = Board(
            square_m=self.square.value() / 1000, marker_m=self.square.value() * 0.7 / 1000
        )
        self.say("Samples and calibration reset. Saved files remain.")

    def save_board(self):
        b = self.current_board()
        path, _ = QFileDialog.getSaveFileName(self, "Save board", "charuco.png", "PNG (*.png)")
        if path:
            b.write(path)
            self.say(
                f"Print board area {b.square_m * 7000:g} × {b.square_m * 5000:g} mm. Verify square {b.square_m * 1000:g} mm. Mount flat."
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


def launch(session=None, screenshot=None):
    app = QApplication.instance() or QApplication([])
    win = Window()
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
