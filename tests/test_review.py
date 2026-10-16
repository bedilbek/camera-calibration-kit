"""Calibration review uses selected observations and preserves the source result."""

import json
from pathlib import Path

import cv2
import numpy as np

from calibration_demo.calibration import Board, detect
from calibration_demo.review import Frame, Review


def test_refit_split_and_save_selected_images(tmp_path):
    review = Review.__new__(Review)
    review.root = tmp_path
    review.board = Board(square_m=0.03, marker_m=0.022)
    review.size = (1280, 720)
    K = np.array([[1050.0, 0, 640], [0, 1040, 360], [0, 0, 1]])
    d = np.zeros(5)
    review.saved = {
        "board": vars(review.board),
        "camera_indices": [1],
        "cameras": [{"K": K.tolist(), "d": [d.tolist()], "metadata": {"size": [1280, 720]}}],
    }
    points = review.board.create().getChessboardCorners()
    ids = np.arange(len(points))
    rng = np.random.default_rng(4)
    review.frames = []
    for i in range(24):
        rotation = np.array([0.13 * np.sin(i), 0.15 * np.cos(i), 0.05 * np.sin(i * 0.3)])
        translation = np.array([0.06 * ((i % 6) - 2.5), 0.05 * ((i // 6) - 1.5), 0.8])
        projected = cv2.projectPoints(points, rotation, translation, K, d)[0].reshape(-1, 2)
        observed = (projected + rng.normal(0, 0.08, projected.shape)).astype(np.float32)
        review.frames.append(
            Frame(Path(f"{i + 1:03d}.png"), ids, observed, np.zeros((720, 1280, 3), np.uint8), 1.0)
        )
    selected = {frame.path.name for frame in review.frames[:-1]}
    fit = review.fit("5 coefficients (OpenCV default)", selected)
    split = review.split_check(fit.model, selected)
    assert len(fit.names) == 23
    assert fit.rms < 0.2
    assert split is not None
    assert np.isfinite(split["ray_grid_px"]).all()
    assert review.coverage(selected).sum() == 23 * len(ids)
    output = review.save(fit, selected, split)
    data = json.loads(output.read_text())
    assert data["review"]["excluded_images"] == ["024.png"]
    assert data["cameras"][0]["metadata"]["views"] == 23
    assert data["review"]["source"] == "intrinsics.json"


def test_another_charuco_shape_and_dictionary_detects():
    board = Board(8, 6, 0.035, 0.025, "DICT_5X5_100")
    image = board.create().generateImage((1200, 900))
    ids, pixels = detect(image, board)
    assert len(ids) == (board.squares_x - 1) * (board.squares_y - 1)
    assert pixels.shape == (len(ids), 2)
