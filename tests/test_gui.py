"""Qt interaction checks. Use QT_QPA_PLATFORM=offscreen on headless hosts."""

import json
import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from calibration_demo.cli import synthetic
from calibration_demo.geometry import project
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


def test_single_camera_solve_save_load_and_stereo_guard(window, monkeypatch):
    rig = window.rig
    window.single_camera.setChecked(True)
    assert not window.ids[1].isEnabled()
    assert all(not button.isEnabled() for button in window.stereo_controls)
    assert window.camera_labels[0].text() == "Camera"
    assert window.start_button.text() == "Start camera"
    assert window.capture_left_button.text() == "Capture board photo"
    assert all(not button.isVisible() for button in window.stereo_controls)
    window.square.setValue(30)
    window.marker.setValue(22)
    window.reset_samples()
    board_points = window.board.create().getChessboardCorners()
    ids = np.arange(len(board_points))
    rng = np.random.default_rng(3)
    for _ in range(22):
        rotation = cv2.Rodrigues(rng.uniform(-0.5, 0.5, 3))[0]
        translation = rng.uniform([-0.4, -0.3, 1.8], [0.4, 0.3, 3.5])
        xyz = (rotation @ board_points.T).T + translation
        window.observations[0].append((ids, project(rig, xyz, 0).astype(np.float32)))
    window.capture_ids = ["usb:046d:0825:AB12CD34"]
    window.verified_capture_ids.add(0)
    window.solve_intrinsics()
    path = window.root / "intrinsics.json"
    data = json.loads(path.read_text())
    assert len(data["cameras"]) == 1
    assert data["board"]["square_m"] == pytest.approx(0.03)
    assert data["board"]["marker_m"] == pytest.approx(0.022)
    assert data["cameras"][0]["metadata"]["views"] == 22
    assert data["camera_identities"] == ["usb:046d:0825:AB12CD34"]
    assert data["camera_identity_source"] == "preview_confirmed_at_capture"
    np.testing.assert_allclose(window.intrinsics[0][0], rig.K1, atol=0.2)
    assert window.intrinsics[1] is None
    window.reset_samples()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(path), ""))
    window.load_intrinsics()
    assert window.square.value() == pytest.approx(30)
    assert window.marker.value() == pytest.approx(22)
    np.testing.assert_allclose(window.intrinsics[0][0], rig.K1, atol=0.2)
    window.latest = ([np.zeros((720, 1280, 3), np.uint8)], [])
    with pytest.raises(ValueError, match="two cameras"):
        window.capture_pair()


def test_single_camera_start_only_requests_one_device(window, monkeypatch):
    from calibration_demo import gui

    class Connection:
        def connect(self, *_):
            pass

    class Worker:
        frames = Connection()
        failure = Connection()

        def __init__(self, indices, size):
            self.indices = indices

        def start(self):
            pass

        def requestInterruption(self):
            pass

        def wait(self, *_):
            return True

    monkeypatch.setattr(gui, "CaptureWorker", Worker)
    monkeypatch.setattr(
        gui,
        "usb_devices",
        lambda: [{"id": "usb:046d:0825:AB12CD34"}],
    )
    window.single_camera.setChecked(True)
    window.ids[0].setValue(1)
    assert window.usb_ids[0].text() == "usb:046d:0825:AB12CD34"
    window.start()
    assert window.worker.indices == [1]
    assert window.capture_ids == ["usb:046d:0825:AB12CD34"]
    assert not window.single_camera.isEnabled()
    assert not window.usb_ids[0].isEnabled()
    window.on_frames(([np.zeros((720, 1280, 3), np.uint8)], []))
    assert window.usb_confirms[0].isEnabled()
    with pytest.raises(ValueError, match="Preview matches ID"):
        window.capture_board(0)
    assert window.stop()
    assert not window.ids[1].isEnabled()


def test_custom_charuco_board_controls(window):
    window.single_camera.setChecked(True)
    window.squares_x.setValue(8)
    window.squares_y.setValue(6)
    window.square.setValue(35)
    window.marker.setValue(25)
    window.dictionary.setCurrentText("DICT_5X5_100")
    window.reset_samples()
    board = window.current_board()
    assert (board.squares_x, board.squares_y) == (8, 6)
    assert board.square_m == pytest.approx(0.035)
    assert board.marker_m == pytest.approx(0.025)
    assert board.dictionary == "DICT_5X5_100"
