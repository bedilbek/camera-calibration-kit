"""Observer offsets, relative-pose convention and perspective framing."""

import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from calibration_demo.geometry import Rig, project
from calibration_demo.view_geometry import (
    camera_outline,
    ground_grid,
    ground_projection,
    look_at,
    observer_pose,
    rig_frame,
)


@pytest.fixture
def rig():
    K = np.array([[700.0, 0, 480], [0, 700, 320], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.array([0.0, -0.25, 0.0]))
    center = np.array([0.8, 0, 0.15])
    return Rig(K, np.zeros(5), K.copy(), np.zeros(5), R, -R @ center, (960, 640), {})


def test_lens_centers_and_metric_observer_offsets(rig):
    centers, up, forward = rig_frame(rig)
    np.testing.assert_allclose(centers[1], [0.8, 0, 0.15], atol=1e-12)
    np.testing.assert_allclose(rig.R @ centers[1] + rig.T.reshape(3), 0, atol=1e-12)
    eye, _, _ = observer_pose(rig)
    offset = eye - centers.mean(axis=0)
    assert np.dot(offset, up) == pytest.approx(1.1)
    assert np.dot(offset, forward) == pytest.approx(-1.1)
    assert np.dot(up, forward) == pytest.approx(0)
    target = np.array([0.2, -0.1, 2.0])
    rotation = look_at(eye, target, up)
    local = rotation @ (target - eye)
    np.testing.assert_allclose(local[:2], 0, atol=1e-12)
    assert local[2] > 0
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)


def test_camera_frustum_corners_match_calibrated_image(rig):
    for camera in (0, 1):
        points = camera_outline(rig, camera)
        np.testing.assert_allclose(points[0], rig_frame(rig)[0][camera], atol=1e-12)
        np.testing.assert_allclose(
            project(rig, points[1:], camera),
            [[0, 0], [959, 0], [959, 639], [0, 639]],
            atol=1e-9,
        )


def test_perspective_fit_offsets_and_missing_marker_lifecycle(rig, tmp_path):
    from PySide6.QtWidgets import QApplication

    from calibration_demo.live_gui import LiveWindow

    app = QApplication.instance() or QApplication([])
    window = LiveWindow()
    window.resize(1280, 900)
    window.view.set_rig(rig)
    window.show()
    app.processEvents()
    view = window.view
    points = np.array([[-0.1, 0, 1.8], [0.6, 0, 1.8]])
    view.set_result({"points_left_frame_m": points, "length_m": 0.7}, (2, 0))
    eye, _ = view.pose()
    for point in view.scene_points():
        pixel = view.screen(point)
        assert pixel is not None
        assert 0 < pixel.x() < view.width()
        assert 60 < pixel.y() < view.height() - 35
    view.yaw = 0.3
    view.fit()
    np.testing.assert_allclose(view.pose()[0], eye)
    assert window.observer_distance.value() == 110
    window.observer_distance.setValue(140)
    centers, up, forward = rig_frame(rig)
    np.testing.assert_allclose(view.pose()[0], centers.mean(axis=0) - 1.4 * forward + 1.4 * up)
    window.observer_distance.setValue(110)
    app.processEvents()
    assert window.grab().save(str(tmp_path / "observer-view.png"))
    view.set_result(None, (2, 0))
    assert view.result is None and view.rig is rig
    view.set_rig(rig)
    assert view.center is None and view.result is None
    window.close()


def test_ground_height_footprints_and_grid_follow_shared_value(rig):
    centers, up, _ = rig_frame(rig)
    points = np.array([[0.2, -0.3, 1.8], [0.5, 0.4, 2.0]])
    feet, heights = ground_projection(rig, points)
    np.testing.assert_allclose(heights, [1.4, 0.7])
    np.testing.assert_allclose((feet - centers[0]) @ up, -1.1)
    np.testing.assert_allclose(points - feet, heights[:, None] * up)
    new_feet, new_heights = ground_projection(rig, points, 1.4)
    np.testing.assert_allclose(new_heights - heights, 0.3)
    np.testing.assert_allclose(new_feet - feet, np.tile(-0.3 * up, (2, 1)))
    for markers in (points, []):
        corners, lines, step_m = ground_grid(rig, markers)
        np.testing.assert_allclose((corners - centers[0]) @ up, -1.1)
        assert step_m == 0.5
        assert 2 <= len(lines) <= 84
    _, _, step_m = ground_grid(rig, [[100, 0, 100]])
    assert step_m > 0.5
