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


def observer_pose(rig, behind_m=0.7, above_m=0.7, yaw=0.0, pitch=0.0):
    centers, up, forward = rig_frame(rig)
    right = unit(np.cross(forward, up))
    offset = -behind_m * forward + above_m * up
    # Orbit the observer around the lens midpoint; zero angles restore the metric preset.
    rotation, _ = cv2.Rodrigues(up * yaw)
    offset = rotation @ offset
    rotation, _ = cv2.Rodrigues((rotation @ right) * pitch)
    return centers.mean(axis=0) + rotation @ offset, up, forward


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
