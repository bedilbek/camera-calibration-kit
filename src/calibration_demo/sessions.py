"""Compare and combine single-camera intrinsic sessions."""

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .camera_identity import recorded_id
from .review import MODELS, Review

MODEL_NAMES = {str(count): name for count, name in zip((4, 5, 8), MODELS)}


def _model_name(model):
    try:
        return MODEL_NAMES[str(model)]
    except KeyError as exc:
        raise ValueError("Choose model 4, 5, or 8") from exc


def _image_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pair(first, second):
    left, right = Review(first), Review(second)
    if left.size != right.size:
        raise ValueError(f"Image sizes differ: {left.size} and {right.size}")
    if left.board != right.board:
        raise ValueError(
            "Board settings differ. Use identical square, marker and dictionary settings."
        )
    return left, right


def _selected(review):
    return {frame.path.name for frame in review.frames if frame.ids is not None}


def _summary(review, repeats, model):
    names = _selected(review)
    hashes = {
        frame.path.name: _image_hash(frame.path)
        for frame in review.frames
        if frame.path.name in names
    }
    input_digest = hashlib.sha256(
        json.dumps(
            {
                "images": hashes,
                "board": vars(review.board),
                "size": review.size,
                "model": model,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    fit = review.fit(model, names)
    split = review.split_check(model, names)
    if split is None:
        raise ValueError("Each session needs at least 20 detected images for a split check")
    deviation = 0.0
    for _ in range(repeats - 1):
        other = review.fit(model, names)
        deviation = max(
            deviation,
            abs(fit.rms - other.rms),
            abs(fit.p95 - other.p95),
            float(np.max(np.abs(fit.K - other.K))),
            float(np.max(np.abs(fit.d - other.d))),
            max(abs(fit.per_view[name] - other.per_view[name]) for name in fit.names),
        )
    counts = review.coverage(names)
    return fit, {
        "session": str(review.root),
        "camera_id": recorded_id(review.saved),
        "selected_images": len(names),
        "rejected_images": len(review.frames) - len(names),
        "fit_rms_px": fit.rms,
        "held_out_rms_px": split["held_out_rms_px"],
        "split_ray_median_px": split["ray_median_px"],
        "split_ray_max_px": split["ray_max_px"],
        "coverage_cells": int(np.count_nonzero(counts)),
        "coverage_cells_with_10_corners": int(np.count_nonzero(counts >= 10)),
        "K": fit.K.tolist(),
        "d": fit.d.tolist(),
        "repeat_count": repeats,
        "repeat_max_parameter_change": deviation,
        "repeat_deterministic_1e_9": deviation <= 1e-9,
        "input_digest_sha256": input_digest,
        "image_sha256": hashes,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
    }


def compare(first, second, repeats=2, model="5"):
    if repeats < 2:
        raise ValueError("Use at least two fits to check repeat determinism")
    model = _model_name(model)
    left, right = _load_pair(first, second)
    fit_a, a = _summary(left, repeats, model)
    fit_b, b = _summary(right, repeats, model)
    identity = (
        "match"
        if a["camera_id"] and a["camera_id"] == b["camera_id"]
        else "different"
        if a["camera_id"] and b["camera_id"]
        else "unverified"
    )
    shared_image_count = len(set(a["image_sha256"].values()) & set(b["image_sha256"].values()))
    width, height = left.size
    grid = np.array(
        [(x, y) for y in np.linspace(0, height - 1, 5) for x in np.linspace(0, width - 1, 9)],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    reference = np.asarray(left.saved["cameras"][0]["K"], dtype=np.float64)
    rays_a = cv2.undistortPoints(grid, fit_a.K, fit_a.d, P=reference).reshape(-1, 2)
    rays_b = cv2.undistortPoints(grid, fit_b.K, fit_b.d, P=reference).reshape(-1, 2)
    distances = np.linalg.norm(rays_a - rays_b, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("Cannot compare the ray maps across the full image")
    return {
        "model": model,
        "size": list(left.size),
        "board": vars(left.board),
        "camera_identity": identity,
        "shared_image_count": shared_image_count,
        "first": a,
        "second": b,
        "fx_difference_percent": 100 * (fit_b.K[0, 0] / fit_a.K[0, 0] - 1),
        "fy_difference_percent": 100 * (fit_b.K[1, 1] / fit_a.K[1, 1] - 1),
        "cx_difference_px": float(fit_b.K[0, 2] - fit_a.K[0, 2]),
        "cy_difference_px": float(fit_b.K[1, 2] - fit_a.K[1, 2]),
        "between_session_ray_median_px": float(np.median(distances)),
        "between_session_ray_max_px": float(np.max(distances)),
        "reference_plane": "original K from first session",
        "note": "Repeat fits test numerical determinism. Between-session values test repeatability, not ground-truth accuracy.",
    }


def mix(first, second, output, assume_same_camera=False, model="5"):
    model = _model_name(model)
    left, right = _load_pair(first, second)
    ids = [recorded_id(left.saved), recorded_id(right.saved)]
    if ids[0] and ids[1] and ids[0] != ids[1]:
        raise ValueError("Recorded camera identities differ. Mixing is blocked.")
    if (not ids[0] or not ids[1]) and not assume_same_camera:
        raise ValueError(
            "Camera identity is missing. Verify both sessions came from the same camera, "
            "then use --assume-same-camera."
        )
    output = Path(output).expanduser().resolve()
    if output.is_relative_to(left.root) or output.is_relative_to(right.root):
        raise ValueError("Put the mixed output outside both source session folders")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    sources = []
    hashes = set()
    combined = []
    for label, review in (("a", left), ("b", right)):
        for frame in review.frames:
            if frame.ids is None:
                continue
            digest = _image_hash(frame.path)
            if digest in hashes:
                raise ValueError(f"Duplicate source image: {frame.path}")
            hashes.add(digest)
            name = f"{label}_{frame.path.name}"
            sources.append(
                {
                    "image": name,
                    "source": str(frame.path),
                    "sha256": digest,
                }
            )
            combined.append((name, frame))
    if len(combined) < 20:
        raise ValueError("The mix needs at least 20 detected images")

    # Fit in memory first. Source sessions and files remain unchanged.
    mixed = Review.__new__(Review)
    mixed.board, mixed.size = left.board, left.size
    mixed.frames = [
        type(frame)(Path(name), frame.ids, frame.pixels, frame.image, frame.focus)
        for name, frame in combined
    ]
    fit = mixed.fit(model, {name for name, _ in combined})
    counts = mixed.coverage(set(fit.names))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".calibration-mix-", dir=output.parent) as temp:
        stage = Path(temp) / "session"
        image_dir = stage / "intrinsic_0"
        image_dir.mkdir(parents=True)
        for item in sources:
            shutil.copy2(item["source"], image_dir / item["image"])
        identity = ids[0] if ids[0] and ids[0] == ids[1] else None
        payload = {
            "board": vars(left.board),
            "camera_indices": [],
            "camera_identities": [identity] if identity else [None],
            "cameras": [
                {
                    "K": fit.K.tolist(),
                    "d": fit.d.tolist(),
                    "metadata": {
                        "size": list(left.size),
                        "views": len(fit.names),
                        "rms_px": fit.rms,
                        "per_view_rms_px": [fit.per_view[name] for name in fit.names],
                        "model": model,
                    },
                }
            ],
            "provenance": {
                "source_sessions": [str(left.root), str(right.root)],
                "identity_basis": "recorded_both" if identity else "manual_override_unverified",
                "source_camera_ids": ids,
                "opencv_version": cv2.__version__,
                "images": sources,
            },
        }
        (stage / "intrinsics.json").write_text(json.dumps(payload, indent=2))
        stage.rename(output)
    return {
        "output": str(output),
        "images": len(combined),
        "fit_rms_px": fit.rms,
        "coverage_cells_with_10_corners": int(np.count_nonzero(counts >= 10)),
        "camera_identity": identity,
        "identity_basis": payload["provenance"]["identity_basis"],
    }
