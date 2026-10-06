import json

import cv2
import numpy as np
import pytest

from calibration_demo.calibration import Board, detect, intrinsic, stereo
from calibration_demo.cli import synthetic
from calibration_demo.geometry import Rig, measure, project, reconstruct


@pytest.fixture
def rig():
    k = np.array([[900.0, 0, 640], [0, 910, 360], [0, 0, 1]])
    r = cv2.Rodrigues(np.array([0.02, 0.14, -0.01]))[0]
    return Rig(
        k,
        np.array([-0.08, 0.02, 0.001, -0.002, 0]),
        k.copy(),
        np.zeros(5),
        r,
        (-r @ np.array([1.0, 0.03, 0.01])).reshape(3, 1),
        (1280, 720),
        {},
    )


def test_convergent_distorted_rig_recovers_metric_segment(rig):
    xyz = np.array([[-0.3, -0.2, 4.0], [0.7, 0.4, 5.0]])
    result = reconstruct(rig, project(rig, xyz, 0), project(rig, xyz, 1))
    np.testing.assert_allclose(result["points_left_frame_m"], xyz, atol=1e-6)
    assert result["length_m"] == pytest.approx(np.linalg.norm(xyz[1] - xyz[0]), abs=1e-6)


def test_bad_correspondence_warns(rig):
    xyz = np.array([[-0.3, -0.2, 4.0], [0.7, 0.4, 5.0]])
    left, right = project(rig, xyz, 0), project(rig, xyz, 1)
    right[0, 1] += 25
    assert reconstruct(rig, left, right)["warnings"]


def test_behind_camera_rejected(rig):
    xyz = np.array([[-0.3, -0.2, -4.0], [0.7, 0.4, -5.0]])
    with pytest.raises(ValueError, match="behind"):
        reconstruct(rig, project(rig, xyz, 0), project(rig, xyz, 1))


def test_invalid_points_and_baseline(rig):
    with pytest.raises(ValueError):
        reconstruct(rig, [[np.nan, 4], [5, 5]], [[1, 1], [4, 4]])
    rig.T[:] = 0
    with pytest.raises(ValueError, match="baseline"):
        rig.validate()


def test_serialization_and_units(rig, tmp_path):
    p = tmp_path / "rig.json"
    rig.save(p)
    np.testing.assert_allclose(Rig.load(p).R, rig.R)
    obj = json.loads(p.read_text())
    obj["units"] = "mm"
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="units"):
        Rig.load(p)


def test_scale_error_is_invisible_to_reprojection(rig):
    xyz = np.array([[-0.3, -0.2, 4.0], [0.7, 0.4, 5.0]])
    left_pixels, r = project(rig, xyz, 0), project(rig, xyz, 1)
    a = reconstruct(rig, left_pixels, r)
    rig.T *= 1.1
    b = reconstruct(rig, left_pixels, r)
    assert b["length_m"] == pytest.approx(a["length_m"] * 1.1)
    assert np.max(b["reprojection_px"]) < 1e-4


def test_synthetic_replay(tmp_path):
    synthetic(tmp_path)
    data = json.loads((tmp_path / "session.json").read_text())
    out = measure(Rig.load(tmp_path / "calibration.json"), *data["points"], data["reference_m"])
    assert out["absolute_error_m"] < 1e-6
    assert out["click_sensitivity"]["valid_samples"] == 200


def test_generated_board_is_detectable(tmp_path):
    b = Board()
    path = tmp_path / "board.png"
    b.write(path)
    im = cv2.copyMakeBorder(
        cv2.imread(str(path)), 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    ids, _ = detect(im, b)
    assert len(ids) == 24


def test_calibration_pipeline_from_diverse_projected_board_views(rig):
    b = Board()
    obj = b.create().getChessboardCorners()
    ids = np.arange(len(obj))
    rng = np.random.default_rng(3)
    left, right, pairs = [], [], []
    for _ in range(22):
        rv = rng.uniform(-0.5, 0.5, 3)
        tv = rng.uniform([-0.4, -0.3, 1.8], [0.4, 0.3, 3.5])
        xyz = (cv2.Rodrigues(rv)[0] @ obj.T).T + tv
        left_pixels = project(rig, xyz, 0).astype(np.float32)
        r = project(rig, xyz, 1).astype(np.float32)
        left.append((ids, left_pixels))
        right.append((ids, r))
        pairs.append(((ids, left_pixels), (ids, r)))
    il = intrinsic(left, rig.size, b)
    ir = intrinsic(right, rig.size, b)
    result = stereo(pairs, rig.size, b, il, ir)
    np.testing.assert_allclose(result.K1, rig.K1, atol=0.2)
    np.testing.assert_allclose(result.T, rig.T, atol=0.001)
    np.testing.assert_allclose(result.R, rig.R, atol=0.001)


def test_partial_charuco_ids_matched(rig):
    b = Board()
    obj = b.create().getChessboardCorners()
    pairs = []
    for i in range(10):
        xyz = obj + np.array([-0.2 + i * 0.01, -0.1, 2 + i * 0.12])
        left_pixels, r = project(rig, xyz, 0), project(rig, xyz, 1)
        il, ir = np.arange(20), np.arange(4, 24)
        pairs.append(((il, left_pixels[il]), (ir, r[ir])))
    meta = {"size": list(rig.size)}
    out = stereo(pairs, rig.size, b, (rig.K1, rig.d1, meta), (rig.K2, rig.d2, meta))
    np.testing.assert_allclose(out.T, rig.T, atol=0.001)
