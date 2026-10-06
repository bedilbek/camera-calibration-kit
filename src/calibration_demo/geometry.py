"""Pure stereo geometry, independent of the GUI and camera hardware."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Rig:
    K1: np.ndarray
    d1: np.ndarray
    K2: np.ndarray
    d2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    size: tuple[int, int]
    metadata: dict

    def validate(self):
        for name in ("K1", "K2", "R"):
            a = getattr(self, name)
            if a.shape != (3, 3) or not np.isfinite(a).all():
                raise ValueError(f"Invalid {name}")
        for k in (self.K1, self.K2):
            if k[0, 0] <= 0 or k[1, 1] <= 0:
                raise ValueError("Focal lengths must be positive")
        if not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(self.R), 1
        ):
            raise ValueError("Invalid rotation")
        if self.T.size != 3 or not np.isfinite(self.T).all() or np.linalg.norm(self.T) < 1e-6:
            raise ValueError("Invalid or zero baseline")
        self.T = self.T.reshape(3, 1)
        if len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError("Invalid image size")
        for d in (self.d1, self.d2):
            if d.size not in (4, 5, 8, 12, 14) or not np.isfinite(d).all():
                raise ValueError("Invalid distortion coefficients")

    def save(self, path):
        self.validate()
        obj = {k: getattr(self, k).tolist() for k in ("K1", "d1", "K2", "d2", "R", "T")}
        obj.update(
            schema_version=1,
            size=list(self.size),
            units="metres",
            convention="X_right = R @ X_left + T",
            metadata=self.metadata,
        )
        Path(path).write_text(json.dumps(obj, indent=2))

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text())
        if obj.get("schema_version") != 1 or obj.get("units") != "metres":
            raise ValueError("Unsupported calibration schema or units")
        if obj.get("convention") != "X_right = R @ X_left + T":
            raise ValueError("Unknown coordinate convention")
        rig = cls(
            **{k: np.asarray(obj[k], dtype=np.float64) for k in ("K1", "d1", "K2", "d2", "R", "T")},
            size=tuple(obj["size"]),
            metadata=obj.get("metadata", {}),
        )
        rig.validate()
        return rig


def project(rig, xyz, camera):
    rv = np.zeros(3) if camera == 0 else cv2.Rodrigues(rig.R)[0]
    tv = np.zeros(3) if camera == 0 else rig.T
    k, d = (rig.K1, rig.d1) if camera == 0 else (rig.K2, rig.d2)
    return cv2.projectPoints(np.asarray(xyz, dtype=float), rv, tv, k, d)[0].reshape(-1, 2)


def reconstruct(rig, left, right):
    rig.validate()
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != (2, 2) or right.shape != (2, 2):
        raise ValueError("Select A then B in each image (four clicks total)")
    for p in (left, right):
        if (
            not np.isfinite(p).all()
            or (p < 0).any()
            or (p[:, 0] >= rig.size[0]).any()
            or (p[:, 1] >= rig.size[1]).any()
        ):
            raise ValueError("Points must be finite and inside the calibrated image")
        if np.linalg.norm(p[1] - p[0]) < 1:
            raise ValueError("Endpoints are too close to select reliably")
    nl = cv2.undistortPoints(left.reshape(-1, 1, 2), rig.K1, rig.d1).reshape(-1, 2)
    nr = cv2.undistortPoints(right.reshape(-1, 1, 2), rig.K2, rig.d2).reshape(-1, 2)
    h = cv2.triangulatePoints(np.c_[np.eye(3), np.zeros(3)], np.c_[rig.R, rig.T], nl.T, nr.T)
    if (np.abs(h[3]) < 1e-10).any():
        raise ValueError("Degenerate geometry: point at infinity")
    xyz = (h[:3] / h[3]).T
    z2 = (rig.R @ xyz.T + rig.T)[2]
    if (xyz[:, 2] <= 0).any() or (z2 <= 0).any():
        raise ValueError("Point is behind a camera; check correspondence and camera order")
    residuals = np.stack(
        [
            np.linalg.norm(project(rig, xyz, 0) - left, axis=1),
            np.linalg.norm(project(rig, xyz, 1) - right, axis=1),
        ]
    )
    a = np.c_[nl, np.ones(2)]
    b = (rig.R.T @ np.c_[nr, np.ones(2)].T).T
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    angles = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    # Acute line angle catches both parallel and antiparallel rays.
    angles = np.minimum(angles, 180 - angles)
    warnings = []
    if residuals.max() > 3:
        warnings.append("Reprojection residual exceeds 3 px; check point matching/calibration")
    if angles.min() < 1:
        warnings.append("Ray angle below 1 degree; depth is poorly constrained")
    return dict(
        length_m=float(np.linalg.norm(xyz[1] - xyz[0])),
        points_left_frame_m=xyz.tolist(),
        reprojection_px=residuals.tolist(),
        ray_angles_deg=angles.tolist(),
        camera_distances_m=np.linalg.norm(xyz, axis=1).tolist(),
        warnings=warnings,
    )


def measure(rig, left, right, reference_m=None, click_sigma_px=1.0):
    result = reconstruct(rig, left, right)
    if not np.isfinite(click_sigma_px) or click_sigma_px <= 0:
        raise ValueError("Click sigma must be positive")
    rng = np.random.default_rng(7)
    lengths = []
    for _ in range(200):
        try:
            trial = reconstruct(
                rig,
                np.asarray(left) + rng.normal(0, click_sigma_px, (2, 2)),
                np.asarray(right) + rng.normal(0, click_sigma_px, (2, 2)),
            )
            lengths.append(trial["length_m"])
        except ValueError:
            pass
    result["click_sensitivity"] = dict(
        sigma_px=click_sigma_px,
        valid_samples=len(lengths),
        length_std_m=float(np.std(lengths)) if lengths else None,
        caveat="Pixel perturbation only; excludes calibration, board scale, timing and lens drift. Not a confidence interval.",
    )
    if reference_m is not None:
        if not np.isfinite(reference_m) or reference_m <= 0:
            raise ValueError("Reference length must be positive")
        err = result["length_m"] - reference_m
        result.update(
            reference_m=reference_m,
            signed_error_m=err,
            absolute_error_m=abs(err),
            absolute_error_percent=100 * abs(err) / reference_m,
        )
    return result
