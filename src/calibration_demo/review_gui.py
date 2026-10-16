"""Interactive review of a saved single-camera calibration session."""

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .review import MODELS, Review


class ImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))

    def show_bgr(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
        self.scene().clear()
        self.scene().addPixmap(QPixmap.fromImage(qimage.copy()))
        self.setSceneRect(0, 0, width, height)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class ReviewWindow(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.review = Review(session)
        self.fit = None
        self.split = None
        self.resize(1500, 940)
        self.setWindowTitle(f"Calibration review | {self.review.root.name}")
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.addWidget(QLabel(f"Session: {self.review.root}  •  Uncheck images, then Refit"))
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Lens model"))
        self.model = QComboBox()
        self.model.addItems(MODELS)
        self.model.setCurrentText("5 coefficients (OpenCV default)")
        controls.addWidget(self.model)
        refit = QPushButton("Refit + compare models")
        refit.clicked.connect(self.refit)
        controls.addWidget(refit)
        save = QPushButton("Save reviewed calibration")
        save.clicked.connect(self.save)
        controls.addWidget(save)
        controls.addStretch()
        layout.addLayout(controls)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.parameters = QLabel()
        self.parameters.setWordWrap(True)
        layout.addWidget(self.parameters)
        splitter = QSplitter()
        self.frames = QTableWidget(0, 6)
        self.frames.setHorizontalHeaderLabels(
            ["Use", "Image", "Error px", "Corners", "Tilt °", "Sharpness"]
        )
        self.frames.horizontalHeader().setStretchLastSection(True)
        self.frames.setMinimumWidth(620)
        self.frames.itemSelectionChanged.connect(self.show_selected)
        splitter.addWidget(self.frames)
        self.tabs = QTabWidget()
        self.views = {}
        for name in (
            "Image + residuals",
            "Undistorted",
            "Coverage",
            "Residual field",
            "Split check",
        ):
            view = ImageView()
            self.views[name] = view
            self.tabs.addTab(view, name)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        self.comparison = QTableWidget(0, 5)
        self.comparison.setHorizontalHeaderLabels(
            ["Model", "Fit RMS px", "Held-out RMS px", "Split ray median px", "Split ray max px"]
        )
        self.comparison.setMaximumHeight(150)
        self.comparison.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.comparison)
        layout.addWidget(
            QLabel(
                "Image tab: green corners and red residual arrows ×25. "
                "Residual field: cell RMS and mean arrows ×100. "
                "Split check compares two interleaved fits; held-out pose is estimated per image. "
                "Sharpness is relative within this session. These are internal checks, not proof of physical accuracy."
            )
        )
        self.setCentralWidget(host)
        self.populate_frames()
        self.refit()

    def populate_frames(self):
        self.frames.blockSignals(True)
        self.frames.setRowCount(len(self.review.frames))
        for row, frame in enumerate(self.review.frames):
            check = QTableWidgetItem()
            check.setFlags(check.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(
                Qt.CheckState.Checked if frame.ids is not None else Qt.CheckState.Unchecked
            )
            self.frames.setItem(row, 0, check)
            self.frames.setItem(row, 1, QTableWidgetItem(frame.path.name))
            for col in range(2, 6):
                self.frames.setItem(row, col, QTableWidgetItem())
        self.frames.blockSignals(False)
        self.frames.selectRow(0)

    def selected_names(self):
        return {
            self.frames.item(row, 1).text()
            for row in range(self.frames.rowCount())
            if self.frames.item(row, 0).checkState() == Qt.CheckState.Checked
        }

    def refit(self):
        names = self.selected_names()
        try:
            fit = self.review.fit(self.model.currentText(), names)
            split = self.review.split_check(self.model.currentText(), names)
        except (ValueError, cv2.error) as exc:
            QMessageBox.warning(self, "Cannot fit these images", str(exc))
            return
        self.fit, self.split = fit, split
        coverage = self.review.coverage(names)
        covered = int(np.count_nonzero(coverage))
        well_sampled = int(np.count_nonzero(coverage >= 10))
        tilt = list(fit.tilts.values())
        split_text = (
            f"Half-split ray disagreement: median {split['ray_median_px']:.1f} px, "
            f"max {split['ray_max_px']:.1f} px."
            + (
                " Add closer, varied board views before relying on this fit."
                if split["ray_median_px"] > 5
                else ""
            )
            if split
            else "Select at least 20 views for the half-split check."
        )
        self.summary.setText(
            f"{len(fit.names)} views | fit RMS {fit.rms:.3f} px | 95th percentile residual "
            f"{fit.p95:.3f} px | grid {covered}/48 cells touched, {well_sampled}/48 with ≥10 corners | "
            f"board tilt {min(tilt):.0f}–{max(tilt):.0f}°. {split_text} "
            "Coverage measures detections, not accuracy in each cell."
        )
        board = self.review.board
        distortion = ", ".join(f"{value:.3g}" for value in fit.d.ravel())
        self.parameters.setText(
            f"Board {board.squares_x}×{board.squares_y}, "
            f"square {board.square_m * 1000:g} mm, marker {board.marker_m * 1000:g} mm, "
            f"{board.dictionary} | fx/fy {fit.K[0, 0]:.1f}/{fit.K[1, 1]:.1f} px | "
            f"cx/cy {fit.K[0, 2]:.1f}/{fit.K[1, 2]:.1f} px | distortion [{distortion}]"
        )
        self.update_frame_table()
        self.views["Coverage"].show_bgr(self.coverage_image(coverage))
        self.views["Residual field"].show_bgr(self.residual_field())
        self.views["Split check"].show_bgr(self.split_image())
        self.show_selected()
        QApplication.processEvents()
        self.compare_models(names)

    def frame_error(self, frame):
        if frame.ids is None:
            return None, None
        if frame.path.name in self.fit.residuals:
            delta = self.fit.residuals[frame.path.name]
            return self.fit.per_view[frame.path.name], delta
        obj = self.review.board.create().getChessboardCorners()[frame.ids].astype(np.float32)
        ok, rotation, translation = cv2.solvePnP(
            obj, frame.pixels.astype(np.float32), self.fit.K, self.fit.d
        )
        if not ok:
            return None, None
        predicted = cv2.projectPoints(obj, rotation, translation, self.fit.K, self.fit.d)[0]
        delta = frame.pixels - predicted.reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))), delta

    def update_frame_table(self):
        self.frames.blockSignals(True)
        for row, frame in enumerate(self.review.frames):
            error, _ = self.frame_error(frame)
            values = (
                "—" if error is None else f"{error:.3f}",
                str(0 if frame.ids is None else len(frame.ids)),
                "—"
                if frame.path.name not in self.fit.tilts
                else f"{self.fit.tilts[frame.path.name]:.1f}",
                f"{frame.focus:.0f}",
            )
            for col, value in enumerate(values, 2):
                self.frames.item(row, col).setText(value)
        self.frames.blockSignals(False)

    def show_selected(self):
        if self.fit is None:
            return
        row = self.frames.currentRow()
        if row < 0:
            return
        frame = self.review.frames[row]
        display = frame.image.copy()
        error, delta = self.frame_error(frame)
        if delta is not None:
            for point, offset in zip(frame.pixels, delta):
                x, y = np.rint(point).astype(int)
                end = tuple(np.rint(point + 25 * offset).astype(int))
                cv2.circle(display, (x, y), 3, (0, 220, 0), -1)
                cv2.arrowedLine(display, (x, y), end, (0, 0, 255), 1, tipLength=0.2)
            label = f"{frame.path.name}: {error:.3f} px RMS | {len(frame.ids)} corners"
        else:
            label = f"{frame.path.name}: {frame.error}"
        cv2.rectangle(display, (0, 0), (660, 40), (20, 20, 20), -1)
        cv2.putText(display, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.views["Image + residuals"].show_bgr(display)
        self.views["Undistorted"].show_bgr(cv2.undistort(frame.image, self.fit.K, self.fit.d))

    def coverage_image(self, counts):
        width, height = self.review.size
        out = np.full((height, width, 3), 245, np.uint8)
        maximum = max(int(counts.max()), 1)
        rows, cols = counts.shape
        for row in range(rows):
            for col in range(cols):
                x0, x1 = round(col * width / cols), round((col + 1) * width / cols)
                y0, y1 = round(row * height / rows), round((row + 1) * height / rows)
                strength = counts[row, col] / maximum
                color = (int(245 - 215 * strength), int(245 - 145 * strength), 245)
                cv2.rectangle(out, (x0, y0), (x1, y1), color, -1)
                cv2.rectangle(out, (x0, y0), (x1, y1), (170, 170, 170), 1)
                cv2.putText(
                    out,
                    str(counts[row, col]),
                    (x0 + 10, y0 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (20, 20, 20),
                    2,
                )
        return out

    def residual_field(self):
        width, height = self.review.size
        rows, cols = 6, 8
        vectors = np.zeros((rows, cols, 2))
        squared = np.zeros((rows, cols))
        counts = np.zeros((rows, cols), dtype=int)
        for frame in self.review.frames:
            if frame.path.name not in self.fit.residuals:
                continue
            for point, delta in zip(frame.pixels, self.fit.residuals[frame.path.name]):
                col = np.clip(int(point[0] * cols / width), 0, cols - 1)
                row = np.clip(int(point[1] * rows / height), 0, rows - 1)
                vectors[row, col] += delta
                squared[row, col] += float(np.dot(delta, delta))
                counts[row, col] += 1
        rms = np.sqrt(squared / np.maximum(counts, 1))
        maximum = max(float(rms.max()), 1e-6)
        out = np.full((height, width, 3), 246, np.uint8)
        for row in range(rows):
            for col in range(cols):
                x0, x1 = round(col * width / cols), round((col + 1) * width / cols)
                y0, y1 = round(row * height / rows), round((row + 1) * height / rows)
                strength = rms[row, col] / maximum
                color = (int(245 - 185 * strength), int(245 - 170 * strength), 245)
                cv2.rectangle(out, (x0, y0), (x1, y1), color, -1)
                cv2.rectangle(out, (x0, y0), (x1, y1), (170, 170, 170), 1)
                x = round((col + 0.5) * width / cols)
                y = round((row + 0.5) * height / rows)
                cv2.circle(out, (x, y), 3, (120, 120, 120), -1)
                if counts[row, col]:
                    end = np.rint([x, y] + 100 * vectors[row, col] / counts[row, col]).astype(int)
                    cv2.arrowedLine(out, (x, y), tuple(end), (0, 0, 160), 2, tipLength=0.25)
                    cv2.putText(
                        out,
                        f"{rms[row, col]:.2f}px n{counts[row, col]}",
                        (x0 + 8, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (30, 30, 30),
                        2,
                    )
        cv2.putText(
            out,
            "Cell RMS colour; mean residual arrow x100; n = corners",
            (20, 32),
            0,
            0.8,
            (0, 0, 0),
            2,
        )
        return out

    def split_image(self):
        width, height = self.review.size
        out = np.full((height, width, 3), 245, np.uint8)
        if self.split is None:
            cv2.putText(out, "Need at least 20 selected images", (40, 80), 0, 1, (0, 0, 0), 2)
            return out
        grid = self.split["ray_grid_px"]
        maximum = max(float(np.max(grid)), 1e-6)
        for row in range(5):
            for col in range(9):
                x0, x1 = round(col * width / 9), round((col + 1) * width / 9)
                y0, y1 = round(row * height / 5), round((row + 1) * height / 5)
                ratio = grid[row, col] / maximum
                color = (int(235 - 215 * ratio), int(235 - 180 * ratio), 245)
                cv2.rectangle(out, (x0, y0), (x1, y1), color, -1)
                cv2.rectangle(out, (x0, y0), (x1, y1), (170, 170, 170), 1)
                cv2.putText(
                    out, f"{grid[row, col]:.1f}", (x0 + 8, y0 + 30), 0, 0.6, (20, 20, 20), 2
                )
        return out

    def compare_models(self, names):
        self.comparison.setRowCount(len(MODELS))
        for row, model in enumerate(MODELS):
            try:
                fit = self.fit if model == self.fit.model else self.review.fit(model, names)
                split = (
                    self.split if model == self.fit.model else self.review.split_check(model, names)
                )
                values = [
                    model,
                    f"{fit.rms:.3f}",
                    "—" if split is None else f"{split['held_out_rms_px']:.3f}",
                    "—" if split is None else f"{split['ray_median_px']:.1f}",
                    "—" if split is None else f"{split['ray_max_px']:.1f}",
                ]
            except (ValueError, cv2.error) as exc:
                values = [model, str(exc), "—", "—", "—"]
            for col, value in enumerate(values):
                self.comparison.setItem(row, col, QTableWidgetItem(value))
        self.comparison.resizeColumnsToContents()

    def save(self):
        if (
            self.fit is None
            or set(self.fit.names) != self.selected_names()
            or self.fit.model != self.model.currentText()
        ):
            QMessageBox.warning(self, "Refit first", "Click Refit after changing images or model.")
            return
        path = self.review.save(self.fit, self.selected_names(), self.split)
        QMessageBox.information(self, "Saved", f"Reviewed calibration saved to:\n{path}")


def launch(session):
    app = QApplication.instance() or QApplication([])
    window = ReviewWindow(session)
    window.show()
    return app.exec()
