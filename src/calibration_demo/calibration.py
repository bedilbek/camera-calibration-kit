"""Metric ChArUco observations and fixed-intrinsic stereo calibration."""

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Rig


@dataclass(frozen=True)
class Board:
    squares_x: int = 7
    squares_y: int = 5
    square_m: float = 0.04
    marker_m: float = 0.028

    def create(self):
        if min(self.squares_x, self.squares_y) < 3 or not 0 < self.marker_m < self.square_m:
            raise ValueError("Board requires >=3 squares each axis and 0 < marker < square")
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_m,
            self.marker_m,
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
        )

    def write(self, path):
        # PNG physical size is not authoritative: verify printed squares using a ruler.
        image = self.create().generateImage(
            (self.squares_x * 240, self.squares_y * 240), marginSize=0
        )
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Cannot save {path}")


def detect(image, board):
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board.create()).detectBoard(image)
    if ids is None or len(ids) < 6:
        raise ValueError("Need at least six visible ChArUco corners")
    return ids.ravel(), corners.reshape(-1, 2)


def _points(board, ids):
    p = board.create().getChessboardCorners()[ids].astype(np.float32)
    if np.linalg.matrix_rank(p[:, :2] - p[:, :2].mean(0), tol=1e-6) < 2:
        raise ValueError("Board corners are collinear")
    return p


def intrinsic(observations, size, board):
    if len(observations) < 10:
        raise ValueError("Capture at least 10 diverse views per camera; aim for 20–30")
    obj = [_points(board, ids) for ids, _ in observations]
    img = [p.astype(np.float32).reshape(-1, 1, 2) for _, p in observations]
    rms, k, d, rv, tv = cv2.calibrateCamera(obj, img, size, None, None)
    errors = [
        float(np.sqrt(np.mean(np.sum((cv2.projectPoints(o, r, t, k, d)[0] - p) ** 2, axis=2))))
        for o, p, r, t in zip(obj, img, rv, tv)
    ]
    return k, d, dict(rms_px=float(rms), per_view_rms_px=errors, views=len(obj), size=list(size))


def stereo(pairs, size, board, left_intr, right_intr):
    if len(pairs) < 8:
        raise ValueError("Capture at least 8 board pairs; aim for 15–20 diverse static poses")
    obj, pl, pr = [], [], []
    for (il, left_pixels), (ir, r) in pairs:
        common, li, ri = np.intersect1d(il, ir, return_indices=True)
        if len(common) < 6:
            continue
        obj.append(_points(board, common))
        pl.append(left_pixels[li].astype(np.float32).reshape(-1, 1, 2))
        pr.append(r[ri].astype(np.float32).reshape(-1, 1, 2))
    if len(obj) < 8:
        raise ValueError("Fewer than 8 valid pairs with six common corner IDs")
    k1, d1, m1 = left_intr
    k2, d2, m2 = right_intr
    if tuple(m1["size"]) != tuple(size) or tuple(m2["size"]) != tuple(size):
        raise ValueError("Image size differs from intrinsic calibration")
    rms, k1, d1, k2, d2, r, t, _, _ = cv2.stereoCalibrate(
        obj,
        pl,
        pr,
        k1.copy(),
        d1.copy(),
        k2.copy(),
        d2.copy(),
        size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7),
    )
    return Rig(
        k1,
        d1,
        k2,
        d2,
        r,
        t,
        size,
        dict(
            stereo_rms_px=float(rms),
            stereo_views=len(obj),
            left=m1,
            right=m2,
            board=vars(board),
            warning="Training RMS is not independent metric validation",
        ),
    )
