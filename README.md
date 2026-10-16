# Calibration Lab

A desktop app for camera calibration. Start with one webcam; add a second camera later for stereo calibration and 3D measurements.

## One camera, two positions (offline-first)

Run `uv run python run.py sequential`. Choose a saved single-camera `intrinsics.json`
for each position (the **same file** is valid when resolution and lens settings are
unchanged). Select or capture a ChArUco board image at position 1, then position 2.
Set **Fixed ChArUco board used for relative pose** independently: square counts X/Y,
measured square edge and marker outer black edge in **mm**, and dictionary. The
extrinsic board may differ in size, layout or dictionary from either intrinsic board.
Count squares along the board's original printed X/Y axes, even if the board is rotated
in the photo. A3/A4 identifies paper size, not checker-square size; verify both edges
with a ruler. Size fields start unset. **Copy position 1 intrinsic board** is an explicit
shortcut only when that board definition matches the actual extrinsic print.
The board must stay fixed between these images. Solve the relative pose and inspect
the common-corner count, board reprojection RMS/p95 and baseline. These fit
diagnostics do not establish physical accuracy.
The solved report and exported JSON record the extrinsic board definition. A wrong
square size scales the baseline and all measured lengths while leaving pixel residuals
almost unchanged; do not choose a size just to make a test distance match.

Next select or capture images containing at least two AprilTags with the same IDs
visible at each position. Any IDs in the supplied family are accepted. Each image's
detected IDs and the shared IDs are shown below the inputs. With two shared IDs,
the pair is selected automatically; with more, choose the **Marker pair to measure**.
The independent reference distance must correspond to that selected pair.
**View detections** opens an image with detected centers and their actual IDs;
this overlay also opens automatically after saving a capture.
For a one-camera run, photograph the board and tags at position 1 before moving
the camera, then photograph both at position 2. Both targets may be in the scene;
the four captures remain separate inputs. If the board is removed before the
position-2 board capture, or the camera is moved between a board and tag capture
at one position, that run cannot use the same relative pose.
The markers must stay fixed between images; each camera position must match its
board image exactly. Use the supplied [marker PDFs](patterns/marker/README.md).
The app detects both tag centers and triangulates their center-to-center distance,
reporting reprojection residuals and ray angles. Compare with a separate tape or
caliper measurement for an independent length error check. Capture buttons use a
single selected camera index and show a preview before saving; saved images can be
loaded in any order. The same geometry code can accept images from two cameras.

Each app launch creates one `sessions/sequential-…` folder for all captures. The
current folder is shown above the inputs; **New session** starts another folder
and clears image selections, results and the reference distance, retaining the
intrinsic selections and extrinsic board settings. Recaptures keep earlier image files. The report save dialog
defaults to the current session folder. Select the intrinsic file before capturing:
capture requests its resolution and blocks saving if the camera returns another size.

If you move the camera after capturing a position's board image, recapture its
board image at the new pose. Keep resolution, zoom and focus fixed. With two fixed
cameras, the board and markers can be photographed at different times without
sharing the scene. With one moving camera, use both targets at each position or
a repeatable mount that restores the exact pose. Each target must be static while
its two position images are taken. Moving people or markers cannot be measured
reliably with sequential captures.

For explanations and saved-session examples, see the [calibration guides](docs/README.md).

To compare two intrinsic captures or combine verified images, see [Compare and combine sessions](docs/session-comparison.md).

## Start here: one webcam

Dependencies are already installed on this Mac. From this folder, run:

```bash
uv run python run.py gui --single-camera --camera 1
```

1. If camera access fails, enable your terminal/Python host in **System Settings → Privacy & Security → Camera**, then restart the app.
2. Check the suggested **Camera USB ID**, then click **Start camera**. Confirm the image comes from that USB webcam and select **Preview matches ID**. If camera index **1** is incorrect, click **Stop**, change **Camera** to **0**, and start again. Use **1280 × 720** throughout calibration.
3. Print [the A4 ChArUco PDF](patterns/calib.io_CHARUCO_A4_297mmx210mm_5x7_30mm_22mm_DICT_4X4_legacy_off_start_id_0.pdf) at **100% / actual size** and mount it flat. Measure a square: it must be **30 mm** (board area **210 × 150 mm**). In the app, set **Board columns 7**, **Rows 5**, **Square 30 mm**, **Marker 22 mm**, and **Dictionary `DICT_4X4_100`** before capturing.
4. Keep the camera fixed. Hold the board still, then click **Capture board photo** once per pose. Collect **20–30 sharp views**: vary tilt, distance, and position across the image, including near its edges. Keep at least six corners visible. The log confirms the view count or reports a detection error.
5. Click **Solve intrinsics**. The app saves images and `intrinsics.json` under `sessions/<timestamp>/`; the exact path and errors appear in the log. A low reprojection error alone does not establish physical accuracy.

## Review a saved calibration

Click **Review intrinsics** in the app and choose the saved session folder, or run `uv run python run.py review` to open the latest one. The review shows each image's corner detections, reprojection vectors, relative sharpness, undistorted preview, image coverage, and two-fit stability. Compare 4-, 5-, and 8-coefficient lens models; uncheck suspect images and click **Refit + compare models** to see the updated diagnostics. **Save reviewed calibration** writes `intrinsics-reviewed.json`; the original `intrinsics.json` and images stay available. Internal fit and split metrics are diagnostics, not an independent physical accuracy test.

For another ChArUco board, enter its printed **columns, rows, square size, marker size, and ArUco dictionary** before collecting images. Match all five values to the board PDF.

Single-camera mode hides the second view and measurement controls. Intrinsics describe this camera's lens; stereo calibration later estimates the relative pose of two cameras. Calibrate each camera individually and keep lens settings/resolution unchanged. Targets must remain stationary.

## Other useful commands

```bash
# Replay an ideal synthetic example without cameras
uv run python run.py gui --session examples/synthetic
# Run tests
uv run --extra dev pytest -q
# Install the locked dependencies in a fresh checkout
uv sync --locked --extra dev
```

## Files you need to know

| Path | Purpose |
|---|---|
| `run.py` | Starts the app without building a package. |
| `src/calibration_demo/` | App code: `gui.py` (interface), `capture.py` (camera), `calibration.py` (board/lens calibration), `geometry.py` (3D), `cli.py` (commands). |
| `examples/` | Printable board and synthetic sample. |
| `sessions/` | Your saved camera images and calibration results. |
| `tests/` | Automated geometry and interface checks. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Python 3.12 and locked, wheel-only dependencies. Source builds are disabled. |

This is a measurement prototype. Synthetic results check the software, not real-camera accuracy.
