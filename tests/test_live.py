"""Live geometry, bounded delivery, capture cleanup and Qt validity lifecycle."""

import json
import os
from time import monotonic

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from calibration_demo.geometry import Rig, project
from calibration_demo.live import LiveWorker, load_live_rig, process_pair


@pytest.fixture
def scene():
    K = np.array([[700.0, 0, 480], [0, 700, 320], [0, 0, 1]])
    rig = Rig(
        K, np.zeros(5), K.copy(), np.zeros(5), np.eye(3), np.array([-0.3, 0, 0]), (960, 640), {}
    )
    xyz = np.array([[-0.08, -0.1, 1.4], [0.25, 0.15, 1.7]])
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    images = []
    for camera in (0, 1):
        image = np.full((640, 960), 255, np.uint8)
        for marker_id, center in zip((0, 1), xyz):
            square = center + np.array(
                [[-0.07, -0.07, 0], [0.07, -0.07, 0], [0.07, 0.07, 0], [-0.07, 0.07, 0]]
            )
            quad = project(rig, square, camera).astype(np.float32)
            marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 200, borderBits=2)
            transform = cv2.getPerspectiveTransform(
                np.float32([[0, 0], [199, 0], [199, 199], [0, 199]]), quad
            )
            warped = cv2.warpPerspective(marker, transform, rig.size, borderValue=255)
            image = np.minimum(image, warped)
        images.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    return rig, xyz, images


def test_real_decoding_and_reconstruction_without_monte_carlo(scene):
    rig, xyz, images = scene
    packet = process_pair(rig, images, (0, 1))
    assert packet["status"] == "Tracking"
    assert packet["result"]["length_m"] == pytest.approx(np.linalg.norm(xyz[1] - xyz[0]), abs=0.006)
    np.testing.assert_allclose(packet["result"]["points_left_frame_m"], xyz, atol=0.012)
    assert "click_sensitivity" not in packet["result"]
    assert process_pair(rig, images, (0, 2))["result"] is None
    blank = np.full_like(images[0], 255)
    assert process_pair(rig, [blank, images[1]], (0, 1))["result"] is None
    with pytest.raises(ValueError, match="resolution"):
        process_pair(rig, [images[0][:100], images[1]], (0, 1))
    with pytest.raises(ValueError, match="different"):
        process_pair(rig, images, (0, 0))


def test_bad_correspondence_suppresses_live_length(scene, monkeypatch):
    rig, xyz, images = scene
    left, right = [dict(zip((0, 1), project(rig, xyz, i))) for i in (0, 1)]
    right[0][1] += 40
    detections = iter((left, right))
    monkeypatch.setattr("calibration_demo.live.detect_tags", lambda *a, **k: next(detections))
    packet = process_pair(rig, images, (0, 1))
    assert packet["result"] is None
    assert "residual" in packet["status"]


def test_live_import_preserves_pose_and_rejects_mismatched_intrinsics(scene, tmp_path):
    rig, _, _ = scene
    rig.save(tmp_path / "calibration.json")
    imported = load_live_rig(tmp_path)
    np.testing.assert_array_equal(imported.T, rig.T)
    payload = {
        "board": {},
        "cameras": [{"K": rig.K1.tolist(), "d": rig.d1.tolist(), "metadata": {"size": rig.size}}],
    }
    path = tmp_path / "intrinsics.json"
    path.write_text(json.dumps(payload))
    load_live_rig(tmp_path, tmp_path, path)
    payload["cameras"][0]["K"][0][0] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="recompute"):
        load_live_rig(tmp_path, path)
    rig.metadata["synthetic"] = True
    rig.save(tmp_path / "calibration.json")
    with pytest.raises(ValueError, match="REAL"):
        load_live_rig(tmp_path)


@pytest.mark.parametrize("failure", ["disconnect", "resolution", "open", "interrupt"])
def test_worker_releases_both_cameras_and_mailbox_is_latest_only(scene, monkeypatch, failure):
    rig, _, images = scene
    worker = LiveWorker((0, 1), rig, (0, 1))
    caps = []

    class Camera:
        def __init__(self, index):
            self.index, self.released, self.count = index, False, 0
            caps.append(self)

        def isOpened(self):
            return failure != "open" or self.index == 0

        def set(self, *_):
            return True

        def grab(self):
            self.count += 1
            return self.count <= 2

        def retrieve(self):
            if failure == "interrupt" and self.index == 1:
                worker.requestInterruption()
            image = images[self.index]
            return True, image[:100] if failure == "resolution" else image

        def release(self):
            self.released = True

    monkeypatch.setattr("calibration_demo.live.cv2.VideoCapture", Camera)
    worker.start()
    assert worker.wait(5000)
    assert len(caps) == 2 and all(cap.released for cap in caps)
    packet = worker.take()
    assert ("error" in packet) == (failure != "interrupt")
    assert worker.take() is None
    worker.publish({"n": 1})
    worker.publish({"n": 2})
    assert worker.take() == {"n": 2}


def test_live_ui_confirmation_missing_and_stale_frames_clear_distance(scene, tmp_path):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from calibration_demo.live_gui import LiveWindow

    app = QApplication.instance() or QApplication([])
    rig, _, images = scene
    window = LiveWindow()
    window.view.set_rig(rig)
    window.show()
    app.processEvents()
    packet = process_pair(rig, images, (0, 1))
    packet.update(images=images, captured_at=monotonic(), fps=30, host_gap_ms=2)
    window.show_packet(packet)
    assert window.view.result is None
    window.confirm.setChecked(True)
    window.show_packet(packet)
    assert window.view.result is not None
    assert window.readout.text() == f"{packet['result']['length_m'] * 100:.0f} cm"
    QTest.mousePress(window.view, Qt.MouseButton.LeftButton, pos=QPoint(200, 150))
    QTest.mouseMove(window.view, QPoint(250, 170))
    QTest.mouseRelease(window.view, Qt.MouseButton.LeftButton, pos=QPoint(250, 170))
    assert window.view.yaw != 0.0
    app.processEvents()
    assert window.grab().save(str(tmp_path / "live-preview.png"))
    window.show_packet({**packet, "result": None, "status": "Missing IDs"})
    assert window.view.result is None and window.readout.text() == "Distance —"
    window.show_packet({**packet, "captured_at": monotonic() - 1})
    assert window.view.result is None
    window.close()
    app.processEvents()


def test_ui_rejects_synthetic_before_opening_devices(tmp_path):
    from PySide6.QtWidgets import QApplication

    from calibration_demo.cli import synthetic
    from calibration_demo.live_gui import LiveWindow

    app = QApplication.instance() or QApplication([])
    synthetic(tmp_path)
    window = LiveWindow(tmp_path)
    window.start()
    assert window.worker is None
    assert "REAL" in window.status.text()
    window.close()
    app.processEvents()
