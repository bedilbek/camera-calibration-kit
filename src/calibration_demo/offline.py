"""Sequential-camera stereo calibration and AprilTag length measurement."""

import json
from pathlib import Path

import cv2
import numpy as np

from .calibration import Board, detect
from .geometry import Rig, measure


def load_intrinsic(path):
    """Return the single-camera calibration, board and its source path."""
    source = Path(path)
    if source.is_dir():
        source = source / "intrinsics.json"
    obj = json.loads(source.read_text())
    if len(obj["cameras"]) != 1:
        raise ValueError("Select a single-camera intrinsic session for each position")
    camera = obj["cameras"][0]
    size = tuple(camera["metadata"]["size"])
    return (
        np.asarray(camera["K"], dtype=np.float64),
        np.asarray(camera["d"], dtype=np.float64),
        size,
        Board(**obj["board"]),
        source,
    )


def read_image(path, size):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if (image.shape[1], image.shape[0]) != size:
        raise ValueError(f"Image {path} has size {image.shape[1::-1]}, expected {size}")
    return image


def _pose(image, board, K, d):
    ids, pixels = detect(image, board)
    object_points = board.create().getChessboardCorners()[ids].astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(object_points, pixels.astype(np.float32), K, d)
    if not ok:
        raise ValueError("Could not solve board pose")
    projected = cv2.projectPoints(object_points, rvec, tvec, K, d)[0].reshape(-1, 2)
    residual = np.linalg.norm(projected - pixels, axis=1)
    rotation = cv2.Rodrigues(rvec)[0]
    return rotation, tvec, ids, residual


def calibrate(left_intrinsic, right_intrinsic, left_board_image, right_board_image, *, board=None):
    """Estimate pose using an independent extrinsic board, fixed in the world.

    Omitted ``board`` retains the original API's intrinsic-board fallback.
    """
    K1, d1, size1, board1, path1 = load_intrinsic(left_intrinsic)
    K2, d2, size2, board2, path2 = load_intrinsic(right_intrinsic)
    if size1 != size2:
        raise ValueError("Intrinsic image resolutions differ")
    board_source = "explicit" if board is not None else "intrinsic_fallback"
    if board is None:
        if board1 != board2:
            raise ValueError("Intrinsic boards differ; supply an explicit extrinsic board")
        board = board1
    board.create()  # Validate the independent board before detection.
    left = read_image(left_board_image, size1)
    right = read_image(right_board_image, size1)
    R1, t1, ids1, error1 = _pose(left, board, K1, d1)
    R2, t2, ids2, error2 = _pose(right, board, K2, d2)
    common = np.intersect1d(ids1, ids2)
    if len(common) < 6:
        raise ValueError("Need at least six common ChArUco corner IDs")
    R = R2 @ R1.T
    T = t2 - R @ t1
    baseline = float(np.linalg.norm(T))
    if baseline < 0.01:
        raise ValueError("Camera positions are too close for reliable depth")
    metadata = {
        "method": "two fixed-board PnP views, one camera may move between positions",
        "left_intrinsic": str(path1),
        "right_intrinsic": str(path2),
        "left_board_image": str(Path(left_board_image)),
        "right_board_image": str(Path(right_board_image)),
        "board": vars(board),
        "board_source": board_source,
        "common_corners": int(len(common)),
        "left_board_rms_px": float(np.sqrt(np.mean(error1**2))),
        "right_board_rms_px": float(np.sqrt(np.mean(error2**2))),
        "left_board_p95_px": float(np.percentile(error1, 95)),
        "right_board_p95_px": float(np.percentile(error2, 95)),
        "baseline_m": baseline,
        "warning": "Board residuals are internal fit diagnostics. Check an independent known length. "
        "Camera positions and lens settings must stay fixed after board capture.",
    }
    rig = Rig(K1, d1, K2, d2, R, T, size1, metadata)
    rig.validate()
    return rig


def detect_tags(image, ids=None):
    """Detect centers of the supplied two-border-bit Kalibr AprilTag prints."""
    parameters = cv2.aruco.DetectorParameters()
    parameters.markerBorderBits = 2
    family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    corners, found_ids, _ = cv2.aruco.ArucoDetector(family, parameters).detectMarkers(image)
    if found_ids is None:
        raise ValueError("No AprilTag 36h11 markers found")
    detections = {}
    for marker_id, quad in zip(found_ids.ravel(), corners):
        marker_id = int(marker_id)
        if marker_id in detections:
            raise ValueError(f"Duplicate marker ID {marker_id}")
        # The projected physical center is not generally the mean of four pixels
        # when the tag is tilted. Map the square's center through its homography.
        square = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        transform = cv2.getPerspectiveTransform(square, quad.reshape(4, 2).astype(np.float32))
        detections[marker_id] = cv2.perspectiveTransform(
            np.asarray([[[0.5, 0.5]]], np.float32), transform
        ).reshape(2)
    if ids is None:
        return detections
    missing = set(ids) - detections.keys()
    if missing:
        raise ValueError(f"Missing marker IDs: {sorted(missing)}; found {sorted(detections)}")
    return np.asarray([detections[i] for i in ids], dtype=np.float64)


def measure_tags(rig, left_image, right_image, ids=None, reference_m=None):
    left_found = detect_tags(read_image(left_image, rig.size))
    right_found = detect_tags(read_image(right_image, rig.size))
    common = sorted(left_found.keys() & right_found.keys())
    if ids is None:
        if len(common) != 2:
            raise ValueError(
                f"Select two shared marker IDs to measure; shared: {common}. "
                f"Position 1: {sorted(left_found)}; position 2: {sorted(right_found)}"
            )
        ids = common
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("Select two different marker IDs")
    if not set(ids).issubset(common):
        raise ValueError(f"Selected marker IDs must appear in both images; shared: {common}")
    left = np.asarray([left_found[i] for i in ids], dtype=np.float64)
    right = np.asarray([right_found[i] for i in ids], dtype=np.float64)
    result = measure(rig, left, right, reference_m)
    result.update(
        marker_ids=list(ids), left_centers_px=left.tolist(), right_centers_px=right.tolist()
    )
    result["detected_marker_ids"] = [sorted(left_found), sorted(right_found)]
    result["note"] = "Center-to-center length. Reference error is the independent accuracy check."
    return result
