"""Review intrinsic calibration from saved camera images, without changing the originals."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibration import Board, detect

MODELS = {
    "4 coefficients (fixed k3)": cv2.CALIB_FIX_K3,
    "5 coefficients (OpenCV default)": 0,
    "8 coefficients (rational)": cv2.CALIB_RATIONAL_MODEL,
}


@dataclass
class Frame:
    path: Path
    ids: np.ndarray | None
    pixels: np.ndarray | None
    image: np.ndarray
    focus: float
    error: str = ""


@dataclass
class Fit:
    model: str
    names: list[str]
    K: np.ndarray
    d: np.ndarray
    rms: float
    per_view: dict[str, float]
    residuals: dict[str, np.ndarray]
    tilts: dict[str, float]
    p95: float


class Review:
    def __init__(self, session):
        self.root = Path(session).expanduser().resolve()
        source = self.root / "intrinsics.json"
        if not source.is_file():
            raise ValueError(f"No intrinsics.json in {self.root}")
        self.saved = json.loads(source.read_text())
        if len(self.saved["cameras"]) != 1:
            raise ValueError("Review currently supports single-camera intrinsic sessions")
        self.board = Board(**self.saved["board"])
        self.size = tuple(self.saved["cameras"][0]["metadata"]["size"])
        directory = self.root / "intrinsic_0"
        paths = sorted(directory.glob("*.png"))
        if not paths:
            raise ValueError(f"No saved camera images in {directory}")
        self.frames = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None or (image.shape[1], image.shape[0]) != self.size:
                raise ValueError(f"Cannot read image at calibrated resolution: {path}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            try:
                ids, pixels = detect(image, self.board)
                x0, y0 = np.floor(pixels.min(axis=0)).astype(int)
                x1, y1 = np.ceil(pixels.max(axis=0)).astype(int)
                region = gray[
                    max(0, y0) : min(image.shape[0], y1 + 1),
                    max(0, x0) : min(image.shape[1], x1 + 1),
                ]
                focus = float(cv2.Laplacian(region, cv2.CV_64F).var())
                self.frames.append(Frame(path, ids, pixels, image, focus))
            except ValueError as exc:
                self.frames.append(Frame(path, None, None, image, focus, str(exc)))

    def fit(self, model, names):
        if model not in MODELS:
            raise ValueError(f"Unknown camera model: {model}")
        selected = [f for f in self.frames if f.path.name in names and f.ids is not None]
        if len(selected) < 10:
            raise ValueError(
                "Select at least 10 detected images; 20–30 varied views are preferable"
            )
        board_points = self.board.create().getChessboardCorners()
        objects = [board_points[f.ids].astype(np.float32) for f in selected]
        pixels = [f.pixels.astype(np.float32).reshape(-1, 1, 2) for f in selected]
        rms, K, d, rotations, translations = cv2.calibrateCamera(
            objects, pixels, self.size, None, None, flags=MODELS[model]
        )
        # OpenCV may pad distortion with inactive zero coefficients.
        d = d[:, :8] if model.startswith("8 coefficients") else d[:, :5]
        if not np.isfinite(rms) or not np.isfinite(K).all() or not np.isfinite(d).all():
            raise ValueError("This image set produced a non-finite camera fit")
        per_view, residuals, tilts = {}, {}, {}
        all_lengths = []
        for frame, obj, observed, rotation, translation in zip(
            selected, objects, pixels, rotations, translations
        ):
            projected = cv2.projectPoints(obj, rotation, translation, K, d)[0].reshape(-1, 2)
            delta = observed.reshape(-1, 2) - projected
            lengths = np.linalg.norm(delta, axis=1)
            name = frame.path.name
            per_view[name] = float(np.sqrt(np.mean(lengths**2)))
            residuals[name] = delta
            normal = cv2.Rodrigues(rotation)[0][:, 2]
            tilts[name] = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1))))
            all_lengths.extend(lengths)
        return Fit(
            model,
            [f.path.name for f in selected],
            K,
            d,
            float(rms),
            per_view,
            residuals,
            tilts,
            float(np.percentile(all_lengths, 95)),
        )

    def coverage(self, names, rows=6, cols=8):
        counts = np.zeros((rows, cols), dtype=int)
        for frame in self.frames:
            if frame.path.name not in names or frame.pixels is None:
                continue
            x = np.clip((frame.pixels[:, 0] * cols / self.size[0]).astype(int), 0, cols - 1)
            y = np.clip((frame.pixels[:, 1] * rows / self.size[1]).astype(int), 0, rows - 1)
            np.add.at(counts, (y, x), 1)
        return counts

    def split_check(self, model, names):
        """Two interleaved fits; pose is re-estimated for each held-out image."""
        ordered = [f.path.name for f in self.frames if f.path.name in names and f.ids is not None]
        first, second = ordered[::2], ordered[1::2]
        if min(len(first), len(second)) < 10:
            return None
        a, b = self.fit(model, first), self.fit(model, second)
        grid = np.array(
            [
                (x, y)
                for y in np.linspace(0, self.size[1] - 1, 5)
                for x in np.linspace(0, self.size[0] - 1, 9)
            ],
            dtype=np.float64,
        ).reshape(-1, 1, 2)
        reference = np.array(self.saved["cameras"][0]["K"])
        rays_a = cv2.undistortPoints(grid, a.K, a.d, P=reference).reshape(-1, 2)
        rays_b = cv2.undistortPoints(grid, b.K, b.d, P=reference).reshape(-1, 2)
        disagreement = np.linalg.norm(rays_a - rays_b, axis=1).reshape(5, 9)
        if not np.isfinite(disagreement).all():
            raise ValueError("Split fits cannot be compared across the full image")
        board_points = self.board.create().getChessboardCorners()
        held_out = []
        for fit, test_names in ((a, second), (b, first)):
            for frame in self.frames:
                if frame.path.name not in test_names:
                    continue
                obj = board_points[frame.ids].astype(np.float32)
                ok, rotation, translation = cv2.solvePnP(
                    obj, frame.pixels.astype(np.float32), fit.K, fit.d
                )
                if not ok:
                    continue
                projected = cv2.projectPoints(obj, rotation, translation, fit.K, fit.d)[0]
                held_out.extend(np.linalg.norm(frame.pixels - projected.reshape(-1, 2), axis=1))
        return {
            "ray_grid_px": disagreement,
            "ray_median_px": float(np.median(disagreement)),
            "ray_max_px": float(np.max(disagreement)),
            "held_out_rms_px": float(np.sqrt(np.mean(np.square(held_out)))),
            "half_focal_px": (float(a.K[0, 0]), float(b.K[0, 0])),
            "half_principal_y_px": (float(a.K[1, 2]), float(b.K[1, 2])),
        }

    def save(self, fit, selected, split):
        """Write a separately named calibration that the app can load later."""
        payload = json.loads(json.dumps(self.saved))
        payload["cameras"][0]["K"] = fit.K.tolist()
        payload["cameras"][0]["d"] = fit.d.tolist()
        payload["cameras"][0]["metadata"] = {
            "size": list(self.size),
            "views": len(fit.names),
            "rms_px": fit.rms,
            "per_view_rms_px": [fit.per_view[n] for n in fit.names],
            "model": fit.model,
        }
        payload["review"] = {
            "selected_images": fit.names,
            "excluded_images": [f.path.name for f in self.frames if f.path.name not in selected],
            "p95_residual_px": fit.p95,
            "coverage_cells": int(np.count_nonzero(self.coverage(selected))),
            "coverage_cells_total": 48,
            "split_held_out_rms_px": None if split is None else split["held_out_rms_px"],
            "split_ray_median_px": None if split is None else split["ray_median_px"],
            "split_ray_max_px": None if split is None else split["ray_max_px"],
            "source": "intrinsics.json",
            "note": "Internal fit diagnostics do not establish independent physical accuracy",
        }
        path = self.root / "intrinsics-reviewed.json"
        path.write_text(json.dumps(payload, indent=2))
        return path
