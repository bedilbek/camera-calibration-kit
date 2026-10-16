# Reviewing Calibration Quality

Run `uv run python run.py review` from the repository root, or click **Review intrinsics** in the app and choose a session folder. To select one explicitly, use `uv run python run.py review sessions/20261014-064430`. The review reads the saved PNGs and board definition; it does not change the original `intrinsics.json`.

## What the numbers and views mean

| Display | What to look for |
|---|---|
| **Fit RMS / per-image Error px** | How closely projected board corners match detected corners in the fitted images. Inspect large per-image errors; a low total RMS alone does not establish accuracy. |
| **Corners, Tilt, Sharpness** | Corner count and board-angle diversity; sharpness is a relative score within this session, not a universal threshold. |
| **Image + residuals** | Green detected corners and red reprojection-error arrows enlarged 25×. Look for blur, bad detections, or consistent error directions. |
| **Undistorted** | A visual check for gross remaining bending of straight scene lines. Training images alone cannot prove performance on new scenes. |
| **Coverage** | A 6×8 grid counting detected corners. A touched cell may still have only one observation; compare the numbers, not just the filled area. |
| **Residual field** | RMS error by image region and mean error direction, with arrows enlarged 100×. Structured arrows or hot regions may indicate model, target, or capture problems. |
| **Split check** | Two interleaved image sets are fitted separately. The coloured 9×5 grid shows how much their inferred viewing rays differ, expressed as pixel displacement. Large differences indicate parameter instability even when both fits have low RMS. |

The bottom table compares three **OpenCV pinhole/Brown-family** choices: four active distortion coefficients (`k3` fixed), five active coefficients (default), and eight active rational coefficients. See [camera-model basics](camera-model-basics.md) for the terms. Extra parameters can lower training RMS while worsening stability. “Held-out RMS” estimates a fresh board pose for each held-out image, so it is an internal check, not fully independent physical validation.

For the equations and limits of all four table values, see [review metrics and equations](review-metrics-math.md).

## Decide which images to keep

Click a row to inspect its corners and blur. Uncheck **Use** for a suspect image, then click **Refit + compare models**; compare error, coverage, and split stability together. Do not discard a difficult edge view solely because its RMS is high: it may provide otherwise missing coverage. **Save reviewed calibration** writes `intrinsics-reviewed.json` with the selected image names and model; the original file and PNGs remain. Click Refit after any selection or model change before saving.

## Example: session `20261014-064430`

The 31-view, 1280×720 A4 session had fit RMS **0.612 px**, but split-ray disagreement was **35.1 px median / 59.7 px maximum**. All 48 coverage cells were touched, while only 28 had at least 10 detected corners. Image `002.png` had the largest per-image RMS, **1.372 px**. Excluding it changed fit RMS to **0.566 px** and split-ray median to **20.2 px**; that is an improvement, but the remaining disagreement is still substantial. The eight-coefficient model barely lowered fit RMS and did not improve split stability. The board corners occupied a relatively small span in many images, so a new set with the board closer, varied tilts, and wider image coverage is the next experiment.

No single score is a universal pass threshold. Stronger evidence comes from repeated calibrations with stable parameters, new images at unused poses and distances, straight-line checks, and the accuracy of the final physical measurement. This review currently does not provide Calib.io's full covariance analysis or other lens families such as fisheye and Double Sphere.
