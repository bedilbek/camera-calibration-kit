"""Qt interaction checks. Use QT_QPA_PLATFORM=offscreen on headless hosts."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from calibration_demo.cli import synthetic
from calibration_demo.gui import Window


@pytest.fixture
def window(tmp_path):
    app = QApplication.instance() or QApplication([])
    synthetic(tmp_path / "sample")
    win = Window(tmp_path / "captures")
    win.show()
    app.processEvents()
    win.read_session(tmp_path / "sample")
    app.processEvents()
    yield win
    win.close()
    app.processEvents()


def test_zoomed_annotation_maps_to_original_pixels(window):
    pane = window.panes[0]
    pane.points = []
    pane.show_image(pane.image)
    pane.scale(1.4, 1.4)
    desired = QPointF(640, 360)
    screen = pane.mapFromScene(desired)
    expected = pane.mapToScene(screen)
    QTest.mouseClick(
        pane.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, screen
    )
    np.testing.assert_allclose(pane.points[0], [expected.x(), expected.y()], atol=1e-6)
    assert np.linalg.norm(np.asarray(pane.points[0]) - [640, 360]) < 3
    QTest.mouseClick(pane.viewport(), Qt.MouseButton.RightButton, pos=screen)
    assert pane.points == []


def test_measure_export_and_reload(window, tmp_path, monkeypatch):
    window.calculate()
    assert "1.7117 m" in window.readout.text()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: str(tmp_path))
    window.export_session()
    saved = next(tmp_path.glob("measurement-*"))
    obj = json.loads((saved / "session.json").read_text())
    assert obj["result"]["length_m"] == pytest.approx(window.report["length_m"])
    window.read_session(saved)
    window.calculate()
    assert window.report["length_m"] == pytest.approx(obj["result"]["length_m"])
    window.clear_points()
    assert window.report is None


def test_synthetic_cannot_start_live_capture(window):
    with pytest.raises(ValueError, match="REAL calibration"):
        window.start()
