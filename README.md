# Calibration Lab

A local Python application: calibrate two cameras, select matching endpoints, reconstruct their 3D positions, and compare their separation against a tape measure.

**Scope:** exactly two USB cameras, stationary targets, standard pinhole lenses with distortion. Converging cameras are supported. This is a demonstration and measurement prototype, not a metrology-certified instrument. No internet is needed after dependencies are installed.

## Quick start

Use CPython **3.12** with [uv](https://docs.astral.sh/uv/). The dependencies are pinned for Intel macOS **13.7.8**: NumPy 2.2.6, OpenCV 4.12.0.88, and PySide6 Essentials 6.11.2. Essentials includes the Qt modules used here without the larger Addons package.

```bash
uv sync --locked --extra dev
PYTHONPATH=src uv run python -m calibration_demo.cli gui --session examples/synthetic
```

`pyproject.toml` sets `no-build = true`: missing compatible wheels cause an error instead of compilation. `package = false` also avoids building the local project; run its CLI directly from `src/`. `uv pip` additionally enforces `only-binary = [":all:"]`. Use uv for project dependencies; ordinary pip does not read these settings. Do not override the policy to install a source-only dependency. `.python-version` selects Python 3.12; `uv.lock` fixes dependency versions. If cache access is restricted, prefix uv commands with `UV_CACHE_DIR=/tmp/calibration-kit-uv-cache`.

The supplied synthetic session has A/B already selected. Press **Measure A–B**. Clear points to practise: **Ctrl+click A then B in each image**, wheel to zoom, drag to pan, right-click to undo. Coordinates are stored in original image pixels regardless of display zoom. On macOS use Control-click as specified, not Command-click.

```bash
PYTHONPATH=src uv run python -m calibration_demo.cli measure examples/synthetic
uv run --extra dev pytest -q
PYTHONPATH=src uv run python -m calibration_demo.cli gui
```

The ideal synthetic target is about **1.7117 m**. It proves the computation works under the modeled conditions; it does not demonstrate real-world accuracy. The app prevents applying synthetic calibration directly to live cameras.

## Camera and board preparation

- Two cameras, two rigid mounts, laptop, tape measure and a flat ChArUco target. Use static high-contrast endpoints, such as marked ends of a rigid pole.
- Start around a 1 m baseline with overlapping views over your intended working distance. Do not force the cameras into a 90-degree cross. A wider baseline trades overlap and correspondence difficulty against better depth sensitivity.
- Print `examples/charuco-7x5-40mm.png` with the **board area exactly 280 × 200 mm**. Each square must measure **40 mm**; markers are 28 mm. Add white outer paper margin and mount flat. Printer “fit to page” can silently change scale; measure it. A3 paper is convenient. If you use a different actual square size, enter that size in the app before any captures.
- Board definition: 7×5 squares, DICT_4X4_100; generated with OpenCV 4.13. Use the supplied generator, not a visually similar board from a different version/configuration.
- Choose two distinct camera indices and a resolution both cameras actually deliver. Defaults are 0, 1 and 1280×720. The app rejects a delivered resolution mismatch.
- Grant camera permission to the terminal/Python host in macOS System Settings; close conferencing software. On Windows check Camera privacy settings. On Linux verify device access and install Qt's normal runtime libraries if missing.
- Lock focus, zoom and any digital crop/stabilization. The app requests autofocus off, but driver support varies. Confirm through the camera's controls. Do not change settings after calibration.

## 1. Intrinsic calibration

1. Set the actual board square length. Start both cameras.
2. Hold the board sharp and still. Capture **left board** repeatedly across diverse tilts, depths and image regions. Repeat for **right board**. The cameras can be calibrated from different poses; they do not need common board visibility for this stage.
3. Aim for **20–30 useful views per camera**, including image corners and tilt about both axes. The software minimum is 10, not an accuracy guarantee. Avoid nearly identical frontal views.
4. Press **Solve intrinsics**. Inspect per-view RMS in the log. The app saves `intrinsics.json` and input captures under `sessions/<timestamp>/`. These are training residuals, not physical measurement error.
5. For bad or blurry samples, use **Reset samples** and recollect. This version does not provide an individual sample editor. Save good intrinsics for future measurement sessions.

## 2. Relative pose, with the final camera placement

1. Position and secure both cameras. Load `intrinsics.json` at its exact resolution.
2. Show the same board to both cameras. Hold it stationary for each capture. Press **Capture stereo board** at **15–20 diverse poses** within the common view. Minimum: 8 accepted pairs, each with at least six common non-collinear ChArUco corners.
3. Press **Solve stereo**. This runs `stereoCalibrate` with `CALIB_FIX_INTRINSIC`, using common corner IDs. `calibration.json` saves both camera models and the relative pose.
4. Inspect RMS and baseline magnitude; compare the baseline roughly with the physical mounting, allowing for camera optical centres. Then validate an independent known segment. Low RMS alone is insufficient.
5. Touching one camera invalidates the relative pose. Moving the whole rigid rig preserves relative pose but changes its world pose.

Suggested counts and warning thresholds are initial defaults, not universal acceptance criteria. Board capture requires both cameras connected in this version, even though intrinsic observations are collected separately.

## 3. Live measurement

1. Both A and B must be stationary and visible in both images. Start with an easy marked object nearby; increase distance only after validation.
2. Press **Freeze / resume** to freeze the image pair.
3. Ctrl+click **A then B in the left image**, and **the same physical A then B in the right image**. Zoom in to pick precise features. Right-click undoes the last point in that image.
4. Enter the independently measured tape length in **metres**, or leave zero for no reference. Press **Measure A–B**.
5. Read the length, 3D endpoints, reprojection residuals in both images, ray angles, and absolute/relative tape error. Camera-to-point distances are also reported separately.
6. **Export session + result** saves original PNGs, calibration, annotations, host timestamps and result JSON in a new folder. Reopen with **Open session** or `PYTHONPATH=src uv run python -m calibration_demo.cli gui --session <folder>`. CLI replay: `PYTHONPATH=src uv run python -m calibration_demo.cli measure <folder>`.

A wrong match can still satisfy epipolar geometry. Inspect the images and independent tape error, not only a reassuring residual. Warnings are advisory; do not present a flagged result as validated.

## What the geometry does

- Stored convention: `X_right = R @ X_left + T`. Left-camera coordinates use x right, y down, z forward. Units are metres. Right camera centre in this frame is `-R.T @ T`.
- Selected distorted pixels pass through `undistortPoints` to normalized coordinates. Triangulation uses `P_left=[I|0]`, `P_right=[R|T]`; no extra K multiplication in this normalized-coordinate path.
- Homogeneous coordinates are divided by their fourth component. Degenerate solutions and points behind either camera are rejected.
- Each reconstructed endpoint is projected back through its camera model, including distortion, to calculate pixel residuals.
- Output is the Euclidean 3D segment length `norm(B-A)`, not image length, floor-plane length or distance along a curved surface.
- Below 1° ray angle or above 3 px maximum reprojection residual triggers an advisory warning. These thresholds need adaptation to your actual rig and required tolerance.
- Click sensitivity perturbs all four clicks by independent Gaussian noise with sigma 1 px over 200 trials. Its reported length standard deviation excludes calibration, physical board scale, timing, lens drift and biased clicking. **It is not a confidence interval or total accuracy estimate.**

## Synchronization and performance

OpenCV grabs each camera before retrieving either image; each rig is owned by a capture worker thread. Host grab timestamps are recorded. **They are not exposure timestamps and do not prove synchronization.** Hold both the board and measured target still. Moving targets require synchronized cameras and an appropriate acquisition pipeline. The UI handles exactly two cameras; extending to N cameras needs a shared rig model, association and multiview optimization.

Calibration solves and short measurement computations run on the UI thread, so the interface can pause while solving. Capture itself runs on a worker. There is no automatic focus verification, calibration sample deletion, bundle adjustment, fisheye model, or hardware-specific camera tuning UI. Test the application on the actual laptop and cameras.

## Why this stack

| Candidate | Suitable use | Decision for this application |
|---|---|---|
| OpenCV 4.12 | ChArUco, USB capture, calibration, triangulation | Core geometry; 4.12 Intel wheels support macOS 13, while 4.13 requires macOS 14 |
| PySide6 Essentials / Qt Graphics View | Native desktop UI; image-space interaction and overlays | Chosen; direct camera access and precise zoom/pan annotation without browser streaming |
| PyQtGraph | Scientific plots, image analysis and interactive ROI tools | Good extension for residual plots; unnecessary dependency for four point clicks |
| Gradio | Fast browser demos with image selection callbacks | Useful for uploaded-image demos; native USB rig is simpler for this application |

`opencv-contrib-python-headless` is deliberate: Qt owns the GUI, so OpenCV does not need a second GUI backend. Do not install multiple packages that provide `cv2` in the same environment. PySide6 is the official Qt binding. Before redistributing a packaged binary, check Qt and dependency license requirements.

## Project structure

- `geometry.py`: validated calibration schema, projection, triangulation and diagnostics.
- `calibration.py`: ChArUco generation/detection, intrinsic and stereo estimation.
- `capture.py`: camera lifecycle and software-paired acquisition worker.
- `gui.py`: capture workflow, image annotation, measurement and session export.
- `cli.py`: GUI launcher, board generation, synthetic session and replay commands.
- `tests/`: numerical geometry, calibration pipeline, board detection, schema and GUI tests.

Dependencies and the wheel-only policy are specified in `pyproject.toml`; `uv.lock` is the installation lockfile. `requirements-tested.txt` records the historical build-validation environment, not this Mac's dependency set or a cross-platform lockfile.

## Official references

- [OpenCV camera calibration and 3D reconstruction](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)
- [OpenCV ChArUco calibration tutorial](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html)
- [OpenCV stereo calibration sample](https://github.com/opencv/opencv/blob/4.x/samples/cpp/stereo_calib.cpp)
- [Qt Graphics View](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QGraphicsView.html)
- [PyQtGraph](https://www.pyqtgraph.org/)
- [Gradio image selection event data](https://www.gradio.app/docs/gradio/selectdata)

Research checked September 2026.
