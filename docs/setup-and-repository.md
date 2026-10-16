# Setup and Repository Map

## Run on this Intel Mac

The project uses CPython 3.12 and packages pinned in `uv.lock`. Dependencies are already installed in `.venv` on the current machine. From the repository root:

```bash
uv run python run.py gui --single-camera --camera 1
uv run python run.py review
uv run --extra dev pytest -q
```

The first command opens live single-camera capture. The second reviews the newest session containing `intrinsics.json`; append a session directory to review a specific one. If webcam 1 shows the wrong device, stop capture and select index 0. Cover the USB webcam briefly to confirm which camera is displayed. macOS camera access may require permission for the terminal or Python host in **System Settings → Privacy & Security → Camera**; restart the app after changing permission.

On a fresh checkout, run `uv sync --locked --extra dev`. If the uv cache is inaccessible, prefix the command with `UV_CACHE_DIR=/tmp/calibration-kit-uv-cache`. The project intentionally requires wheels: `pyproject.toml` sets `no-build = true`, `package = false`, and `uv pip` binary-only behavior. Do not relax those settings to resolve a missing package; check wheel availability for Intel macOS before changing a pin.

## Find a file

| Location | Purpose |
|---|---|
| `run.py`, `src/calibration_demo/cli.py` | User commands and entry point. |
| `src/calibration_demo/gui.py`, `capture.py` | Live camera interface and device acquisition. |
| `src/calibration_demo/calibration.py` | ChArUco detection and intrinsic/stereo fitting. |
| `src/calibration_demo/review.py`, `review_gui.py` | Saved-image refitting, quality metrics, and review interface. |
| `src/calibration_demo/geometry.py` | Calibration schema, projection, triangulation, and measurement. |
| `patterns/` | User-generated calib.io board PDFs. |
| `sessions/<timestamp>/` | Captured PNGs and calibration JSON files. This directory is Git-ignored; back it up separately to preserve an experiment. |
| `tests/`, `examples/` | Automated checks and synthetic demonstration data. |

`Open session` in the main window expects a saved stereo/measurement session. Use **Review intrinsics** or the `review` command for a one-camera `intrinsics.json` session.
