"""Offline-first sequential camera workflow with optional live image capture."""

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
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

from .calibration import Board, detect
from .offline import calibrate, detect_tags, load_intrinsic, measure_tags


class CaptureDialog(QDialog):
    def __init__(self, parent, camera_index, output, size):
        super().__init__(parent)
        self.setWindowTitle("Capture image — verify camera preview")
        self.output = output
        self.size = size
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            self.camera.release()
            raise ValueError(f"Cannot open camera index {camera_index}")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.preview = QLabel("Waiting for camera")
        self.preview.setMinimumSize(640, 360)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame = None
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"Required: {size[0]} × {size[1]}. "
                "Check the intended board or markers and camera position in preview."
            )
        )
        layout.addWidget(self.preview)
        self.save_button = QPushButton("Save this frame")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_frame)
        layout.addWidget(self.save_button)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(50)

    def next_frame(self):
        ok, frame = self.camera.read()
        if not ok or frame is None:
            self.frame = None
            self.save_button.setEnabled(False)
            return
        actual = (frame.shape[1], frame.shape[0])
        if actual != self.size:
            self.frame = None
            self.save_button.setEnabled(False)
            self.preview.setText(
                f"Camera returned {actual[0]} × {actual[1]}; "
                f"calibration requires {self.size[0]} × {self.size[1]}.\n"
                "Close capture and check the camera index and supported resolution."
            )
            self.timer.stop()
            self.camera.release()
            return
        self.frame = frame
        self.save_button.setEnabled(True)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
        self.preview.setPixmap(
            QPixmap.fromImage(image.copy()).scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio
            )
        )

    def save_frame(self):
        if self.frame is None:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.output), self.frame):
            QMessageBox.warning(self, "Capture failed", f"Cannot save {self.output}")
            return
        self.accept()

    def done(self, result):
        self.timer.stop()
        self.camera.release()
        super().done(result)


class OfflineWindow(QMainWindow):
    fields = (
        ("Intrinsic • position 1", "intrinsic1", True),
        ("Intrinsic • position 2", "intrinsic2", True),
        ("Fixed board • position 1", "board1", False),
        ("Fixed board • position 2", "board2", False),
        ("Two tags • position 1", "tags1", False),
        ("Two tags • position 2", "tags2", False),
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sequential stereo • offline and live capture")
        self.resize(1100, 900)
        self.rig = None
        self.result = None
        self.inputs = {}
        root = QWidget()
        layout = QVBoxLayout(root)
        note = QLabel(
            "One camera can visit positions 1 and 2. Keep the board still for both board images. "
            "Keep camera pose, focus, zoom and resolution fixed between board and tag images "
            "at each position. Keep tags still during both tag images."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        session_row = QHBoxLayout()
        self.session_label = QLabel()
        self.session_label.setWordWrap(True)
        self.session_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        session_row.addWidget(self.session_label, 1)
        self.new_session_button = QPushButton("New session")
        self.new_session_button.clicked.connect(lambda: self.guard(self.new_session))
        session_row.addWidget(self.new_session_button)
        layout.addLayout(session_row)
        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera index for capture"))
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 30)
        camera_row.addWidget(self.camera_index)
        camera_row.addStretch()
        layout.addLayout(camera_row)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for label, key, intrinsic in self.fields:
            row = QHBoxLayout()
            field = QLineEdit()
            field.setPlaceholderText(
                "Select a session/intrinsics.json" if intrinsic else "Select image"
            )
            field.textChanged.connect(self.invalidate)
            self.inputs[key] = field
            row.addWidget(field)
            browse = QPushButton("Choose")
            browse.clicked.connect(lambda _=False, k=key, i=intrinsic: self.choose(k, i))
            row.addWidget(browse)
            if not intrinsic:
                capture = QPushButton("Capture")
                capture.clicked.connect(lambda _=False, k=key: self.capture(k))
                row.addWidget(capture)
                preview = QPushButton("View detections")
                preview.clicked.connect(lambda _=False, k=key: self.view(k))
                row.addWidget(preview)
            form.addRow(label, row)
        layout.addLayout(form)
        board_group = QGroupBox("Fixed ChArUco board used for relative pose")
        board_layout = QVBoxLayout(board_group)
        board_note = QLabel(
            "This board may differ from the intrinsic board. Measure the printed square edge "
            "and ArUco marker outer black edge; A3/A4 paper size does not define them. "
            "Use the same fixed board for both positions."
        )
        board_note.setWordWrap(True)
        board_layout.addWidget(board_note)
        shape_row = QHBoxLayout()
        self.board_squares_x = QSpinBox()
        self.board_squares_y = QSpinBox()
        for field, value in ((self.board_squares_x, 7), (self.board_squares_y, 5)):
            field.setRange(3, 100)
            field.setValue(value)
            field.valueChanged.connect(self.invalidate)
        shape_row.addWidget(QLabel("Squares X"))
        shape_row.addWidget(self.board_squares_x)
        shape_row.addWidget(QLabel("Squares Y"))
        shape_row.addWidget(self.board_squares_y)
        self.board_square_mm = QDoubleSpinBox()
        self.board_marker_mm = QDoubleSpinBox()
        for label, field in (
            ("Square edge", self.board_square_mm),
            ("Marker edge", self.board_marker_mm),
        ):
            field.setRange(0, 10000)
            field.setDecimals(3)
            field.setSuffix(" mm")
            field.setSpecialValueText("Enter measured size")
            field.valueChanged.connect(self.invalidate)
            shape_row.addWidget(QLabel(label))
            shape_row.addWidget(field)
        board_layout.addLayout(shape_row)
        dictionary_row = QHBoxLayout()
        dictionary_row.addWidget(QLabel("Dictionary"))
        self.board_dictionary = QComboBox()
        self.board_dictionary.addItems(
            sorted(name for name in dir(cv2.aruco) if name.startswith("DICT_"))
        )
        self.board_dictionary.setCurrentText("DICT_4X4_100")
        self.board_dictionary.currentTextChanged.connect(self.invalidate)
        dictionary_row.addWidget(self.board_dictionary, 1)
        copy_board = QPushButton("Copy position 1 intrinsic board")
        copy_board.clicked.connect(lambda: self.guard(self.copy_intrinsic_board))
        dictionary_row.addWidget(copy_board)
        board_layout.addLayout(dictionary_row)
        layout.addWidget(board_group)
        self.marker_summary = QLabel(
            "Select tag images to see detected marker IDs at each position."
        )
        self.marker_summary.setWordWrap(True)
        layout.addWidget(self.marker_summary)
        marker_row = QHBoxLayout()
        marker_row.addWidget(QLabel("Marker pair to measure"))
        self.marker_pair = QComboBox()
        self.marker_pair.addItem("Select tag images first", None)
        self.marker_pair.currentIndexChanged.connect(self.invalidate)
        marker_row.addWidget(self.marker_pair, 1)
        refresh = QPushButton("Refresh detections")
        refresh.clicked.connect(self.refresh_markers)
        marker_row.addWidget(refresh)
        layout.addLayout(marker_row)
        for key in ("tags1", "tags2"):
            self.inputs[key].textChanged.connect(self.refresh_markers)
        reference_row = QHBoxLayout()
        reference_row.addWidget(QLabel("Independent measured tag-center distance (m)"))
        self.reference = QDoubleSpinBox()
        self.reference.setRange(0, 1000)
        self.reference.setDecimals(5)
        self.reference.setSingleStep(0.001)
        self.reference.setSpecialValueText("Not measured")
        self.reference.valueChanged.connect(self.invalidate)
        reference_row.addWidget(self.reference)
        reference_row.addStretch()
        layout.addLayout(reference_row)
        row = QHBoxLayout()
        solve = QPushButton("1  Solve relative pose + review")
        solve.clicked.connect(lambda: self.guard(self.solve))
        row.addWidget(solve)
        measure = QPushButton("2  Detect tags + measure")
        measure.clicked.connect(lambda: self.guard(self.measure))
        row.addWidget(measure)
        export = QPushButton("Save rig + report")
        export.clicked.connect(lambda: self.guard(self.save))
        row.addWidget(export)
        layout.addLayout(row)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        layout.addWidget(self.report, 1)
        self.setCentralWidget(root)
        self.new_session()

    def new_session(self):
        output = Path("sessions").resolve() / datetime.now(timezone.utc).strftime(
            "sequential-%Y%m%d-%H%M%S-%f"
        )
        output.mkdir(parents=True, exist_ok=False)
        self.session_dir = output
        self.session_label.setText(f"Session: {output}")
        for key in ("board1", "board2", "tags1", "tags2"):
            self.inputs[key].clear()
        self.reference.setValue(0)
        self.invalidate()

    def extrinsic_board(self):
        square_mm = self.board_square_mm.value()
        marker_mm = self.board_marker_mm.value()
        if not 0 < marker_mm < square_mm:
            raise ValueError(
                "Enter the extrinsic board's measured edges: 0 < marker edge < square edge (mm)"
            )
        board = Board(
            squares_x=self.board_squares_x.value(),
            squares_y=self.board_squares_y.value(),
            square_m=square_mm / 1000,
            marker_m=marker_mm / 1000,
            dictionary=self.board_dictionary.currentText(),
        )
        board.create()
        return board

    def copy_intrinsic_board(self):
        board = load_intrinsic(self.value("intrinsic1"))[3]
        self.board_squares_x.setValue(board.squares_x)
        self.board_squares_y.setValue(board.squares_y)
        self.board_square_mm.setValue(board.square_m * 1000)
        self.board_marker_mm.setValue(board.marker_m * 1000)
        self.board_dictionary.setCurrentText(board.dictionary)

    def refresh_markers(self, *_):
        previous = self.marker_pair.currentData()
        found = []
        summaries = []
        for position in (1, 2):
            path = self.inputs[f"tags{position}"].text().strip()
            try:
                image = cv2.imread(path) if path else None
                if image is None:
                    raise ValueError("Select a readable tag image")
                ids = set(detect_tags(image))
                found.append(ids)
                summaries.append(f"Position {position}: IDs {sorted(ids)}")
            except ValueError as exc:
                found.append(set())
                summaries.append(f"Position {position}: {exc}")
        common = sorted(found[0] & found[1])
        self.marker_summary.setText("; ".join(summaries) + f". Shared IDs: {common}")
        self.marker_pair.clear()
        self.marker_pair.addItem("Choose two shared marker IDs", None)
        for pair in combinations(common, 2):
            self.marker_pair.addItem(f"ID {pair[0]} ↔ ID {pair[1]}", pair)
        previous_index = next(
            (
                i
                for i in range(1, self.marker_pair.count())
                if self.marker_pair.itemData(i) == previous
            ),
            None,
        )
        if previous_index is not None:
            self.marker_pair.setCurrentIndex(previous_index)
        elif len(common) == 2:
            self.marker_pair.setCurrentIndex(1)

    def choose(self, key, intrinsic):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select intrinsic calibration" if intrinsic else "Select image",
            "",
            "JSON (*.json)" if intrinsic else "Images (*.png *.jpg *.jpeg *.tif *.tiff)",
        )
        if path:
            self.inputs[key].setText(path)

    def capture(self, key):
        try:
            size = load_intrinsic(self.value(f"intrinsic{key[-1]}"))[2]
            output = (
                self.session_dir / f"{key}-{datetime.now(timezone.utc).strftime('%H%M%S-%f')}.png"
            )
            dialog = CaptureDialog(self, self.camera_index.value(), output, size)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.inputs[key].setText(str(output))
                self.view(key)
        except Exception as exc:
            QMessageBox.warning(self, "Capture", str(exc))

    def view(self, key):
        try:
            image = cv2.imread(self.value(key))
            if image is None:
                raise ValueError("Cannot read selected image")
            try:
                if key.startswith("board"):
                    board = self.extrinsic_board()
                    ids, pixels = detect(image, board)
                    for point in pixels:
                        cv2.drawMarker(
                            image, tuple(point.astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 10, 2
                        )
                    cv2.putText(
                        image,
                        f"{len(ids)} ChArUco corners",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                elif key.startswith("tags"):
                    centers = detect_tags(image)
                    for marker_id, point in centers.items():
                        center = tuple(point.astype(int))
                        cv2.drawMarker(image, center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                        cv2.putText(
                            image,
                            f"ID {marker_id}",
                            (center[0] + 10, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                    cv2.putText(
                        image,
                        f"Detected IDs: {sorted(centers)}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
            except ValueError as exc:
                QMessageBox.information(self, "Detection", str(exc))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
            dialog = QDialog(self)
            dialog.setWindowTitle(key)
            label = QLabel()
            label.setPixmap(
                QPixmap.fromImage(qimage.copy()).scaled(
                    1000, 700, Qt.AspectRatioMode.KeepAspectRatio
                )
            )
            QVBoxLayout(dialog).addWidget(label)
            dialog.exec()
        except Exception as exc:
            QMessageBox.warning(self, "View image", str(exc))

    def guard(self, action):
        try:
            action()
        except Exception as exc:
            QMessageBox.warning(self, "Check workflow", str(exc))

    def invalidate(self, *_):
        self.rig = None
        self.result = None
        self.report.clear()

    def value(self, key):
        value = self.inputs[key].text().strip()
        if not value:
            raise ValueError(f"Select {key} first")
        return value

    def solve(self):
        self.rig = calibrate(
            self.value("intrinsic1"),
            self.value("intrinsic2"),
            self.value("board1"),
            self.value("board2"),
            board=self.extrinsic_board(),
        )
        self.result = None
        m = self.rig.metadata
        self.report.setPlainText(
            f"Extrinsic board: {m['board']['squares_x']} × {m['board']['squares_y']} squares, "
            f"square {m['board']['square_m'] * 1000:g} mm, "
            f"marker {m['board']['marker_m'] * 1000:g} mm, {m['board']['dictionary']}\n"
            f"Relative pose: {m['baseline_m']:.4f} m baseline, {m['common_corners']} common corners\n"
            f"Board reprojection RMS: {m['left_board_rms_px']:.3f} / "
            f"{m['right_board_rms_px']:.3f} px (positions 1 / 2)\n"
            f"Board reprojection p95: {m['left_board_p95_px']:.3f} / "
            f"{m['right_board_p95_px']:.3f} px\n\n{m['warning']}"
        )

    def measure(self):
        if self.rig is None:
            self.solve()
        self.result = measure_tags(
            self.rig,
            self.value("tags1"),
            self.value("tags2"),
            ids=self.marker_pair.currentData(),
            reference_m=self.reference.value() or None,
        )
        result = self.result
        self.report.appendPlainText(
            f"\nTag {result['marker_ids'][0]} to tag {result['marker_ids'][1]}: "
            f"{result['length_m']:.4f} m\n"
            + (
                f"Independent absolute error: {result['absolute_error_m'] * 1000:.2f} mm "
                f"({result['absolute_error_percent']:.2f}%)\n"
                if "absolute_error_m" in result
                else ""
            )
            + f"Reprojection residuals (px): {result['reprojection_px']}\n"
            f"Ray angles (deg): {result['ray_angles_deg']}\n"
            f"Warnings: {result['warnings']}\n"
            "For accuracy, compare this with an independently measured center-to-center length."
        )

    def save(self):
        if self.rig is None:
            raise ValueError("Solve relative pose first")
        directory = QFileDialog.getExistingDirectory(
            self, "Choose output folder", str(self.session_dir)
        )
        if not directory:
            return
        output = Path(directory)
        self.rig.save(output / "calibration.json")
        (output / "review.json").write_text(
            json.dumps(
                {
                    "inputs": {k: v.text() for k, v in self.inputs.items()},
                    "pose": self.rig.metadata,
                    "measurement": self.result,
                },
                indent=2,
            )
        )
        self.report.appendPlainText(f"\nSaved calibration.json and review.json to {output}")


def launch():
    app = QApplication.instance() or QApplication([])
    window = OfflineWindow()
    window.show()
    return app.exec()
