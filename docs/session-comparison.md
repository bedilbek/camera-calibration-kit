# Compare and Combine Intrinsic Sessions

Run these commands from the repository root. Keep the original session folders.

## Compare two sessions

```bash
uv run python run.py compare-sessions sessions/20261014-064430 sessions/20261014-081710
uv run python run.py compare-sessions sessions/20261014-064430 sessions/20261014-081710 --model 4
uv run python run.py compare-sessions sessions/20261014-064430 sessions/20261014-081710 --json
```

The tool checks that image size and board settings match. It redetects corners and fits the selected camera model to each session. Model 5 is the default; use `--model 4` or `--model 8` to compare alternatives. It reports fit RMS, held-out RMS, split-ray disagreement, image coverage, and camera-matrix differences. The between-session ray values compare two models on a common 9 × 5 image grid. Read [review metrics and equations](review-metrics-math.md) for the definitions.

The tool fits each unchanged image set twice by default. It reports an input SHA-256 digest and software versions in JSON output. It also counts identical image files shared by both sessions. A maximum numerical change of at most \(10^{-9}\) means the repeated software calculation was deterministic in this run. It does **not** mean two new capture sessions will produce the same calibration. Between-session differences measure repeatability. They do not measure error against a true camera model.

For these two saved sessions, the 41-image session has fit RMS **0.518 px**, held-out RMS **0.528 px**, and split-ray median **21.6 px**. The 31-image session has **0.612 px**, **0.625 px**, and **35.1 px**. Well-sampled coverage rises from **28/48** to **37/48** cells. The fitted focal lengths still differ by about **2–3%**. The two ray maps differ by **17.1 px median** on the comparison grid. This suggests improved coverage and internal stability, but does not prove physical accuracy. Camera identity is unverified in both old files.

On the 41-image session, model 4 has fit RMS **0.532 px** and split-ray median **11.0 px**. Model 5 has fit RMS **0.518 px** and split-ray median **21.6 px**. Model 8 has a split-ray maximum of **83.2 px**. The lower fit error of the flexible models does not make them the better choice by itself.

## Record the physical camera

```bash
uv run python run.py camera-id
```

On this Mac, the Logitech device reports vendor ID `046d`, product ID `0825`, and a serial number. The Lenovo hub has different IDs. A USB webcam normally has no network MAC address. Use the full USB ID printed by `camera-id`. Do not use the USB port location or the OpenCV camera index as a permanent camera ID.

For a new single-camera session, the app suggests an ID when exactly one serial-bearing non-hub USB device is connected. The suggestion is not proof that this device supplies the image. Start the camera, check the live preview, and select **Preview matches ID** before board capture. Clear or replace the suggestion if it is wrong. The app checks that the named USB device is connected and saves the confirmed ID with `intrinsics.json`. It cannot map an OpenCV index to a USB ID. With two cameras, enter and confirm a different ID for each preview.

The two existing sessions have no recorded ID. Label one only if you confirmed which physical camera supplied its live preview during capture:

```bash
uv run python run.py tag-session sessions/20261014-064430 usb:046d:0825:SERIAL_FROM_CAMERA_ID --verified-from-preview
uv run python run.py tag-session sessions/20261014-081710 usb:046d:0825:SERIAL_FROM_CAMERA_ID --verified-from-preview
```

The tool marks this as a manual, after-capture identification. It cannot recover proof that was not saved during capture. Do not label a session if its physical camera is uncertain.

## Make one fit from both sessions

Mix images only if the same physical camera, resolution, board, focus, zoom, and lens settings were used. A mix can improve the fit if the new images add useful board positions and tilts. It can also hide a change in camera settings.

Mixing stops when recorded IDs differ. If an ID is missing, first verify the camera and add `--assume-same-camera` to the mix command. This override does not permit conflicting IDs. The tool also blocks different board settings, resolutions, duplicate image files, and an existing output folder.

After you verify that the two old sessions used the same camera, run:

```bash
uv run python run.py mix-sessions sessions/20261014-064430 sessions/20261014-081710 --output sessions/combined-4 --model 4 --assume-same-camera
uv run python run.py review sessions/combined-4
```

The output has copied PNGs, a new `intrinsics.json`, and source paths plus SHA-256 hashes for every image. The source sessions remain unchanged. The output records whether both session files supplied the same ID or a manual override was used. An override does not add a verified ID to the mixed result. Review the mixed result against each separate fit; a lower fit RMS alone does not prove that the mix is better.

A disposable test mix of these 72 images with model 4 gave **0.574 px fit RMS** and **19.0 px split-ray median**. The second session alone gave **0.532 px** and **11.0 px** with model 4. The mix does not improve these internal checks. For any saved mix, inspect `provenance.identity_basis` before treating its camera ID as verified.
