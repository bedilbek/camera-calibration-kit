"""Sequential one-camera geometry and real AprilTag decoding."""

import json

import cv2
import numpy as np
import pytest

from calibration_demo.calibration import Board
from calibration_demo.geometry import project
from calibration_demo.offline import calibrate, detect_tags, measure_tags


@pytest.mark.parametrize("explicit_board", [False, True])
def test_sequential_pose_and_length(tmp_path, monkeypatch, explicit_board):
    board = Board(square_m=0.03, marker_m=0.022)
    intrinsic_board = Board(squares_x=6, square_m=0.015, marker_m=0.011)
    size = (1280, 720)
    K = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
    d = np.zeros(5)
    intr = tmp_path / "intrinsics.json"
    intr.write_text(
        json.dumps(
            {
                "board": vars(intrinsic_board if explicit_board else board),
                "cameras": [{"K": K.tolist(), "d": d.tolist(), "metadata": {"size": size}}],
            }
        )
    )
    for name in ("board1", "board2", "tags1", "tags2"):
        assert cv2.imwrite(str(tmp_path / f"{name}.png"), np.zeros((720, 1280, 3), np.uint8))
    R = cv2.Rodrigues(np.array([0.0, 0.35, 0.0]))[0]
    camera2 = np.array([0.8, 0, 0])
    T = -R @ camera2
    board_points = board.create().getChessboardCorners()
    ids = np.arange(len(board_points))
    board_rotation = cv2.Rodrigues(np.array([0.25, -0.2, 0.05]))[0]
    world = (board_rotation @ board_points.astype(float).T).T + np.array([-0.15, -0.08, 2.5])
    from calibration_demo.geometry import Rig

    expected = Rig(K, d, K, d, R, T, size, {})
    observations = iter([(ids, project(expected, world, 0)), (ids, project(expected, world, 1))])

    def observed(image, detected_board):
        assert detected_board == board
        return next(observations)

    monkeypatch.setattr("calibration_demo.offline.detect", observed)
    right_intr = intr
    if explicit_board:
        payload = json.loads(intr.read_text())
        payload["board"] = vars(Board(squares_x=8, dictionary="DICT_5X5_100"))
        right_intr = tmp_path / "right-intrinsics.json"
        right_intr.write_text(json.dumps(payload))
    rig = calibrate(
        intr,
        right_intr,
        tmp_path / "board1.png",
        tmp_path / "board2.png",
        **({"board": board} if explicit_board else {}),
    )
    np.testing.assert_allclose(rig.R, R, atol=1e-4)
    np.testing.assert_allclose(rig.T.ravel(), T, atol=1e-4)
    assert rig.metadata["common_corners"] == len(ids)
    assert rig.metadata["left_board_rms_px"] < 1e-3
    assert rig.metadata["board"] == vars(board)
    assert rig.metadata["board_source"] == ("explicit" if explicit_board else "intrinsic_fallback")
    tags = np.array([[0, 0, 3], [0.3, 0.2, 3.2]])
    centers = iter([project(expected, tags, 0), project(expected, tags, 1)])
    monkeypatch.setattr(
        "calibration_demo.offline.detect_tags",
        lambda image: dict(zip((0, 2), next(centers))),
    )
    length = float(np.linalg.norm(tags[1] - tags[0]))
    result = measure_tags(rig, tmp_path / "tags1.png", tmp_path / "tags2.png", reference_m=length)
    assert result["length_m"] == pytest.approx(length, abs=1e-5)
    assert result["absolute_error_m"] < 1e-5
    assert result["marker_ids"] == [0, 2]


def test_detect_two_border_bit_tags():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    image = np.full((400, 800), 255, np.uint8)
    for marker_id, x in ((0, 100), (1, 450)):
        tag = cv2.aruco.generateImageMarker(dictionary, marker_id, 200, borderBits=2)
        image[100:300, x : x + 200] = tag
    centers = detect_tags(image)
    assert set(centers) == {0, 1}
    np.testing.assert_allclose([centers[0], centers[1]], [[199.5, 199.5], [549.5, 199.5]], atol=1)
    np.testing.assert_allclose(detect_tags(image, (1, 0)), [centers[1], centers[0]])


def test_offline_window_has_independent_inputs(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from calibration_demo.offline_gui import OfflineWindow

    app = QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    window = OfflineWindow()
    assert len(window.inputs) == 6
    assert set(window.inputs) == {"intrinsic1", "intrinsic2", "board1", "board2", "tags1", "tags2"}
    window.close()
    app.processEvents()


def test_extrinsic_board_fields_units_copy_and_invalidation(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from calibration_demo import offline_gui

    app = QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    window = offline_gui.OfflineWindow()
    with pytest.raises(ValueError, match="measured edges"):
        window.extrinsic_board()
    intrinsic_board = Board(square_m=0.03, marker_m=0.022)
    monkeypatch.setattr(
        offline_gui, "load_intrinsic", lambda path: (None, None, (1280, 720), intrinsic_board)
    )
    window.inputs["intrinsic1"].setText("intrinsics.json")
    window.copy_intrinsic_board()
    assert window.extrinsic_board() == intrinsic_board
    window.board_square_mm.setValue(52)
    window.board_marker_mm.setValue(38)
    window.board_squares_x.setValue(5)
    window.board_squares_y.setValue(7)
    window.board_dictionary.setCurrentText("DICT_5X5_100")
    expected = Board(5, 7, 0.052, 0.038, "DICT_5X5_100")
    assert window.extrinsic_board() == expected
    window.rig = object()
    window.result = {"length_m": 1}
    window.report.setPlainText("stale result")
    window.board_square_mm.setValue(53)
    assert window.rig is None and window.result is None
    assert not window.report.toPlainText()
    window.board_square_mm.setValue(52)
    window.new_session()
    assert window.extrinsic_board() == expected
    window.board_marker_mm.setValue(52)
    with pytest.raises(ValueError, match="measured edges"):
        window.extrinsic_board()
    window.close()
    app.processEvents()


def test_sequential_session_groups_captures_and_preserves_previous_session(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QDialog

    from calibration_demo import offline_gui

    app = QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    window = offline_gui.OfflineWindow()
    viewed = []
    monkeypatch.setattr(window, "view", viewed.append)
    first_session = window.session_dir
    assert first_session.is_dir()
    window.inputs["intrinsic1"].setText("calibration1.json")
    window.inputs["intrinsic2"].setText("calibration2.json")
    requested = []

    class FakeDialog:
        def __init__(self, parent, camera_index, output, size):
            requested.append((output, size))
            output.write_bytes(b"saved frame")

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(offline_gui, "CaptureDialog", FakeDialog)
    monkeypatch.setattr(offline_gui, "load_intrinsic", lambda path: (None, None, (1280, 720)))
    for key in ("board1", "board2", "tags1", "tags2", "board1"):
        window.capture(key)
    assert all(path.parent == first_session and size == (1280, 720) for path, size in requested)
    assert len(set(path for path, _ in requested)) == 5
    assert viewed == ["board1", "board2", "tags1", "tags2", "board1"]
    window.reference.setValue(0.4)
    window.new_session_button.click()
    assert window.session_dir != first_session
    assert window.session_dir.is_dir()
    assert len(list(first_session.glob("*.png"))) == 5
    assert all(not window.inputs[key].text() for key in ("board1", "board2", "tags1", "tags2"))
    assert window.inputs["intrinsic1"].text() == "calibration1.json"
    assert window.reference.value() == 0
    window.close()
    app.processEvents()


def test_marker_pair_ui_lists_real_ids_and_shared_pairs(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from calibration_demo.offline_gui import OfflineWindow

    app = QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for position, ids in ((1, (0, 2, 5, 7)), (2, (2, 5, 7, 9))):
        image = np.full((400, 1400), 255, np.uint8)
        for index, marker_id in enumerate(ids):
            x = 50 + index * 330
            image[100:300, x : x + 200] = cv2.aruco.generateImageMarker(
                dictionary, marker_id, 200, borderBits=2
            )
        assert cv2.imwrite(str(tmp_path / f"tags{position}.png"), image)
    window = OfflineWindow()
    for position in (1, 2):
        window.inputs[f"tags{position}"].setText(str(tmp_path / f"tags{position}.png"))
    assert "Position 1: IDs [0, 2, 5, 7]" in window.marker_summary.text()
    assert "Position 2: IDs [2, 5, 7, 9]" in window.marker_summary.text()
    assert window.marker_pair.currentData() is None
    assert [window.marker_pair.itemData(i) for i in range(1, 4)] == [(2, 5), (2, 7), (5, 7)]
    window.marker_pair.setCurrentIndex(3)
    window.refresh_markers()
    assert window.marker_pair.currentData() == (5, 7)
    window.new_session()
    assert window.marker_pair.count() == 1
    window.close()
    app.processEvents()


def test_measure_tags_requires_shared_pair_and_explicit_choice_for_multiple(monkeypatch):
    from types import SimpleNamespace

    from calibration_demo import offline

    monkeypatch.setattr(offline, "read_image", lambda path, size: path)
    detections = {
        "left": {0: [1, 2], 2: [3, 4], 5: [5, 6]},
        "right": {0: [7, 8], 2: [9, 10], 5: [11, 12], 9: [13, 14]},
    }
    monkeypatch.setattr(offline, "detect_tags", lambda image: detections[image])
    rig = SimpleNamespace(size=(1280, 720))
    with pytest.raises(ValueError, match="Select two shared"):
        measure_tags(rig, "left", "right")
    with pytest.raises(ValueError, match="must appear in both"):
        measure_tags(rig, "left", "right", ids=(2, 9))
    monkeypatch.setattr(offline, "measure", lambda rig, left, right, ref: {"length_m": 0.4})
    result = measure_tags(rig, "left", "right", ids=(2, 5))
    assert result["marker_ids"] == [2, 5]
    assert result["left_centers_px"] == [[3, 4], [5, 6]]
    assert result["right_centers_px"] == [[9, 10], [11, 12]]


@pytest.mark.parametrize("height, can_save", [(720, True), (960, False)])
def test_sequential_capture_enforces_calibrated_resolution(tmp_path, monkeypatch, height, can_save):
    from PySide6.QtWidgets import QApplication, QDialog

    from calibration_demo import offline_gui

    app = QApplication.instance() or QApplication([])
    settings = {}

    class FakeCamera:
        released = False

        def isOpened(self):
            return True

        def set(self, prop, value):
            settings[prop] = value
            return True

        def read(self):
            return True, np.zeros((height, 1280, 3), np.uint8)

        def release(self):
            self.released = True

    camera = FakeCamera()
    monkeypatch.setattr(offline_gui.cv2, "VideoCapture", lambda index: camera)
    output = tmp_path / "frame.png"
    dialog = offline_gui.CaptureDialog(None, 0, output, (1280, 720))
    assert not dialog.save_button.isEnabled()
    dialog.next_frame()
    assert settings[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert settings[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert dialog.save_button.isEnabled() == can_save
    dialog.save_frame()
    assert output.exists() == can_save
    if can_save:
        assert cv2.imread(str(output)).shape[:2] == (720, 1280)
    else:
        assert "1280 × 960" in dialog.preview.text()
    dialog.done(QDialog.DialogCode.Rejected)
    assert camera.released
    app.processEvents()
