# One-Camera Intrinsic Calibration

Intrinsic calibration estimates this camera's focal lengths (`fx`, `fy`), principal point (`cx`, `cy`), and lens distortion. It is tied to a camera, resolution, focus/zoom state, and image-processing mode. Another webcam of the same model still needs its own calibration.

## Capture a session

1. Print and measure the [A4 ChArUco board](boards-and-printing.md). Connect the USB webcam and close other apps using it.
2. Run `uv run python run.py gui --single-camera --camera 1`. Confirm the **Camera** index shows the USB camera. Keep **Width 1280** and **Height 720** if that is the intended working resolution.
3. Enter **Board columns 7**, **Rows 5**, **Square 30 mm**, **Marker 22 mm**, and **Dictionary `DICT_4X4_100`**. Click **Start camera**.
4. Hold the board still and click **Capture board photo** once. This saves **one still image**, not a video. Move and tilt the board, wait for a sharp frame, then click again. Collect about **20–30 distinct views**. The app requires at least 10 and reports successful captures in the log.
5. Include board positions near the centre, all image edges, and corners. Vary distance and tilt about both axes, including a few rotated views. Bring the board closer so its detected corners span a useful fraction of the frame, while keeping enough of it visible for detection. Avoid many almost identical front-facing views.
6. Click **Solve intrinsics**. The app writes `sessions/<timestamp>/intrinsic_0/001.png`, subsequent numbered images, and `intrinsics.json`. The latter stores the board definition, camera matrix `K`, distortion vector `d`, image size, view count, and per-view reprojection errors.

Keep the camera fixed throughout the capture set. Verify that focus is stable; the code requests autofocus off, but device support varies. If a capture fails, read the log, improve lighting/sharpness or reveal more board corners, and try that pose again. `Freeze / resume` is for stereo point selection; leave live capture running while collecting board views.

After solving, open [the quality-review guide](reviewing-calibration.md). A successful solve and low RMS establish only a fit to the captured images. For a clean second attempt, start a **new app session**; `Reset samples` clears memory but does not delete earlier files in the same session folder.
