"""Stress-test supplied Kalibr marker PDFs in simulated camera views.

Run with ``.venv/bin/python tools/evaluate_kalibr_markers.py`` on macOS.
Uses built-in ``sips`` and the project's pinned OpenCV. This digital test
cannot establish printed-camera accuracy.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ROOT / "patterns" / "marker"
IMAGE_SIZE = (1280, 720)
TAG_SIZE_M = 0.1
FOCAL_PX = 900.0


def detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.markerBorderBits = 2  # Kalibr's generator uses two black border bits.
    family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.ArucoDetector(family, params)


def render_pdf(path: Path, output: Path) -> np.ndarray:
    subprocess.run(
        ["sips", "-s", "format", "png", str(path), "--out", str(output)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    image = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not render {path}")
    return image


def detections(image: np.ndarray, tag_detector: cv2.aruco.ArucoDetector) -> list[dict]:
    corners, ids, _ = tag_detector.detectMarkers(image)
    if ids is None:
        return []
    return [
        {"id": int(tag_id), "corners_px": quad.reshape(4, 2)}
        for tag_id, quad in zip(ids.ravel(), corners)
    ]


def projected_quad(
    width_px: float, yaw_deg: float, pitch_deg: float, rng: np.random.Generator
) -> np.ndarray:
    yaw, pitch = np.deg2rad([yaw_deg, pitch_deg])
    rotate_y = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rotate_x = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]]
    )
    corners_m = np.array([[-0.05, -0.05, 0], [0.05, -0.05, 0], [0.05, 0.05, 0], [-0.05, 0.05, 0]])
    depth_m = FOCAL_PX * TAG_SIZE_M / width_px
    points = corners_m @ (rotate_y @ rotate_x).T + [0, 0, depth_m]
    center_px = np.array([640, 360]) + rng.uniform(-120, 120, 2)
    return (points[:, :2] / points[:, 2:3] * FOCAL_PX + center_px).astype(np.float32)


def simulate(
    image: np.ndarray,
    source_quad: np.ndarray,
    target_quad: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    homography = cv2.getPerspectiveTransform(source_quad, target_quad)
    camera = cv2.warpPerspective(
        image, homography, IMAGE_SIZE, flags=cv2.INTER_LINEAR, borderValue=255
    )
    camera = cv2.GaussianBlur(camera, (3, 3), 0.65)
    noise = rng.normal(0, 3, camera.shape)
    return np.clip(camera.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def projected_center(quad: np.ndarray) -> np.ndarray:
    unit = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    transform = cv2.getPerspectiveTransform(unit, quad)
    return cv2.perspectiveTransform(np.float32([[[50, 50]]]), transform)[0, 0]


def run_sweep() -> None:
    rng = np.random.default_rng(20261014)
    tag_detector = detector()
    files = [
        MARKERS / f"calib.io_KALIBR_A4_210x297_1x1_100_t36h11_startid_{i}.pdf" for i in range(4)
    ]
    with tempfile.TemporaryDirectory() as temp:
        images = [render_pdf(path, Path(temp) / f"marker_{i}.png") for i, path in enumerate(files)]
    source_detection = [detections(image, tag_detector) for image in images]
    if any(len(found) != 1 for found in source_detection):
        raise RuntimeError("Expected exactly one AprilTag per PDF")
    source_quads = [found[0]["corners_px"] for found in source_detection]
    source_ids = [found[0]["id"] for found in source_detection]
    report = {
        "pdf_ids_by_filename": list(range(4)),
        "decoded_pdf_ids": source_ids,
        "simulated_camera": {"size_px": IMAGE_SIZE, "focal_px": FOCAL_PX},
        "scenarios": [],
        "negative_frames": 0,
        "negative_false_detections": 0,
        "paired_trials": 0,
        "paired_correct": 0,
        "paired_unexpected_ids": 0,
    }
    for width_px in (20, 30, 40, 60, 80, 120):
        for tilt_deg in (0, 30, 50, 65):
            correct = mismatched = missed = 0
            center_errors = []
            for file_id, image in enumerate(images):
                for _ in range(8):
                    yaw_deg = tilt_deg * (1 if rng.random() > 0.5 else -1)
                    pitch_deg = rng.uniform(-10, 10)
                    target = projected_quad(width_px, yaw_deg, pitch_deg, rng)
                    camera = simulate(image, source_quads[file_id], target, rng)
                    found = detections(camera, tag_detector)
                    if not found:
                        missed += 1
                        continue
                    match = next((item for item in found if item["id"] == file_id), None)
                    if match is None:
                        mismatched += 1
                        match = found[0]
                    else:
                        correct += 1
                    center_errors.append(
                        float(
                            np.linalg.norm(
                                projected_center(match["corners_px"]) - projected_center(target)
                            )
                        )
                    )
            report["scenarios"].append(
                {
                    "width_px": width_px,
                    "yaw_deg": tilt_deg,
                    "trials": correct + mismatched + missed,
                    "correct": correct,
                    "mismatched": mismatched,
                    "missed": missed,
                    "median_center_error_px": (
                        float(np.median(center_errors)) if center_errors else None
                    ),
                }
            )
    for _ in range(100):
        background = rng.integers(170, 256, size=(720, 1280), dtype=np.uint8)
        for _ in range(3):
            x, y = rng.integers(50, 1150), rng.integers(50, 600)
            side = rng.integers(20, 100)
            cv2.rectangle(background, (int(x), int(y)), (int(x + side), int(y + side)), 0, -1)
        report["negative_frames"] += 1
        report["negative_false_detections"] += len(detections(background, tag_detector))
    for _ in range(50):
        left_quad = projected_quad(80, 30, rng.uniform(-10, 10), rng)
        right_quad = projected_quad(80, -30, rng.uniform(-10, 10), rng)
        left_quad[:, 0] -= 300
        right_quad[:, 0] += 300
        left = simulate(images[0], source_quads[0], left_quad, rng)
        right = simulate(images[1], source_quads[1], right_quad, rng)
        found_ids = [item["id"] for item in detections(np.minimum(left, right), tag_detector)]
        report["paired_trials"] += 1
        report["paired_correct"] += set(found_ids) == {0, 1}
        report["paired_unexpected_ids"] += len(set(found_ids) - {0, 1})
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", help="Inspect a real camera frame")
    args = parser.parse_args()
    if not args.image:
        run_sweep()
        return
    tag_detector = detector()
    result = {}
    for path in args.image:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            parser.error(f"Could not read image: {path}")
        result[str(path)] = [
            {
                "id": item["id"],
                "center_px": projected_center(item["corners_px"]).tolist(),
                "corners_px": item["corners_px"].tolist(),
            }
            for item in detections(image, tag_detector)
        ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
