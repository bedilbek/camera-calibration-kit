"""Metric observer and camera outlines in the left camera coordinate system."""

import cv2
import numpy as np


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if length < 1e-8:
        raise ValueError("Cannot define observer direction from this camera arrangement")
    return vector / length


def rig_frame(rig):
    """Assume level cameras: average optical up defines the ground normal."""
    centers = np.array([np.zeros(3), -rig.R.T @ rig.T.reshape(3)])
    up = unit(np.array([0, -1, 0]) + rig.R.T @ np.array([0, -1, 0]))
    forward = np.array([0, 0, 1]) + rig.R.T @ np.array([0, 0, 1])
    forward = unit(forward - up * np.dot(forward, up))
    return centers, up, forward


def observer_pose(rig, behind_m=1.1, above_m=1.1, yaw=0.0, pitch=0.0):
    centers, up, forward = rig_frame(rig)
    right = unit(np.cross(forward, up))
    offset = -behind_m * forward + above_m * up
    # Orbit the observer around the lens midpoint; zero angles restore the metric preset.
    rotation, _ = cv2.Rodrigues(up * yaw)
    offset = rotation @ offset
    rotation, _ = cv2.Rodrigues((rotation @ right) * pitch)
    return centers.mean(axis=0) + rotation @ offset, up, forward


def ground_projection(rig, points, camera_height_m=1.1):
    """Project onto the assumed ground, anchored below the left camera lens."""
    centers, up, _ = rig_frame(rig)
    origin = centers[0] - camera_height_m * up
    points = np.asarray(points, dtype=float)
    heights = (points - origin) @ up
    return points - heights[..., None] * up, heights


def ground_grid(rig, points, camera_height_m=1.1):
    """Bounded ground patch enclosing the camera and marker footprints."""
    centers, up, forward = rig_frame(rig)
    right = unit(np.cross(forward, up))
    origin = centers[0] - camera_height_m * up
    points = np.vstack([centers, np.asarray(points).reshape(-1, 3)])
    local = (points - origin) @ np.array([right, forward]).T
    lower = np.floor((local.min(axis=0) - 0.25) * 2) / 2
    upper = np.ceil((local.max(axis=0) + 0.25) * 2) / 2

    def world(x, z):
        return origin + x * right + z * forward

    corners = np.array(
        [
            world(lower[0], lower[1]),
            world(upper[0], lower[1]),
            world(upper[0], upper[1]),
            world(lower[0], upper[1]),
        ]
    )
    step_m = max(0.5, float(np.ceil(np.max(upper - lower) / 40 * 2) / 2))
    lines = []
    for x in np.arange(lower[0], upper[0] + 1e-8, step_m):
        lines.append((world(x, lower[1]), world(x, upper[1])))
    for z in np.arange(lower[1], upper[1] + 1e-8, step_m):
        lines.append((world(lower[0], z), world(upper[0], z)))
    return corners, lines, step_m


def look_at(eye, target, up):
    forward = unit(np.asarray(target) - eye)
    right = unit(np.cross(forward, up))
    return np.array([right, np.cross(right, forward), forward])


def camera_outline(rig, camera, depth_m=0.25):
    """Lens and four undistorted corner rays, for an illustrative frustum depth."""
    K, d = (rig.K1, rig.d1) if camera == 0 else (rig.K2, rig.d2)
    w, h = rig.size
    pixels = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)
    rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, d).reshape(-1, 2)
    points = np.vstack([np.zeros(3), np.c_[rays, np.ones(4)] * depth_m])
    if camera == 1:
        points = (rig.R.T @ (points - rig.T.reshape(3)).T).T
    return points
