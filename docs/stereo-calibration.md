# Adding a Second Camera

**Intrinsics** describe each camera's imaging geometry. **Extrinsics** describe the fixed position and orientation of camera 2 relative to camera 1. In this repository, the left camera is the reference frame and the saved convention is `X_right = R @ X_left + T`, with distances in metres. Two webcams of the same model still require separate intrinsics.

## Current app workflow

1. Mount both cameras rigidly with overlapping views at the intended working distance. Keep resolution, focus, zoom, and camera order fixed. Use a board that both cameras can see clearly.
2. Run `uv run python run.py gui` and leave **Single camera (intrinsics)** unchecked. Choose distinct **Left camera** and **Right camera** indices. Enter the printed ChArUco board parameters and start both cameras.
3. Collect about **20–30** diverse **Capture left board** observations and **20–30 Capture right board** observations. These intrinsic sets need not show the board simultaneously. Click **Solve intrinsics**; this saves one `intrinsics.json` containing both cameras.
4. With the final camera mounting unchanged, show the same stationary board to **both cameras** at each pose. Click **Capture stereo board** at **15–20** diverse common-view poses; at least 8 valid pairs are required. Then click **Solve stereo**. The app writes `calibration.json` with both intrinsics and the relative `R`, `T`.
5. Check the reported baseline against the approximate physical camera separation. For a real accuracy check, measure a rigid segment of known length at several positions and depths. Freeze a live stereo pair, **Ctrl-click A then B in each image**, enter the tape measurement in metres, and click **Measure A–B**. Compare the reconstructed length and reprojection residuals with the known length.

**Current limitation:** the dual-camera GUI expects both camera intrinsics in the *same* two-camera `intrinsics.json`. It cannot directly combine two separately saved single-camera JSON files, so plan a fresh dual-camera intrinsic capture. Camera frames are paired in software; host timestamps do not guarantee simultaneous exposure. Keep the calibration board and measured object still when capturing pairs. A stereo RMS is also a training score; independent length checks are needed before relying on 3D measurements.
