# Calibration Lab Guides

These guides are a durable reference for the workflow, controls, and calibration-quality checks discussed during development. Start with the one-camera guide; use the two-camera guide only after the first calibration has been reviewed.

| Guide | When to read it |
|---|---|
| [Setup and repository map](setup-and-repository.md) | Install or troubleshoot the Intel Mac environment; find the relevant code and saved files. |
| [Boards and printing](boards-and-printing.md) | Choose a calib.io PDF, print to scale, and enter matching board parameters. |
| [One-camera intrinsic calibration](intrinsic-calibration.md) | Capture useful views and produce `intrinsics.json`. |
| [Camera models and metric scale](camera-model-basics.md) | Understand `K`, distortion terms, overfitting, and why print dimensions matter. |
| [Reviewing calibration quality](reviewing-calibration.md) | Interpret the visualizations, compare models, exclude images, and understand the October 14 example. |
| [Review metrics and equations](review-metrics-math.md) | See how Fit RMS, held-out RMS, and split-ray values are calculated. |
| [Compare and combine sessions](session-comparison.md) | Check repeatability, record a camera USB ID, and refit a guarded image mix. |
| [Adding a second camera](stereo-calibration.md) | Calibrate both cameras, solve their relative pose, and validate a 3D measurement. |

The quick command is `uv run python run.py gui --single-camera --camera 1`. To reopen the latest intrinsic review, run `uv run python run.py review`. **`sessions/` is ignored by Git**, so copy it separately if you want to revisit the original images and calculations on another machine.
