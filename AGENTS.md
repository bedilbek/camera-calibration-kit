# Repository Guidelines

## Project Structure & Module Organization

`src/calibration_demo/` contains single-camera intrinsic calibration and two-camera 3D measurement:
- `geometry.py`: calibration schema, projection, triangulation, and diagnostics.
- `calibration.py`: board generation/detection and intrinsic/stereo estimation.
- `capture.py`: camera lifecycle and acquisition worker.
- `gui.py` and `cli.py`: desktop workflow and command-line entry points.

`run.py` launches the app. `tests/` holds geometry and Qt tests; `examples/` holds board and synthetic assets. `README.md` documents setup and calibration.

## Build, Test, and Development Commands

Use CPython 3.12 with uv; `.python-version` selects it and `uv.lock` pins dependencies.

```bash
uv sync --locked --extra dev
uv run python run.py gui --single-camera --camera 1
```

- `uv run python run.py measure examples/synthetic`: replay measurement as JSON. Use `gui --session examples/synthetic` for the sample GUI, or `synthetic --output /tmp/calibration-sample` for disposable data.
- `uv run --extra dev pytest -q`: run all tests.
- `uv run --extra dev ruff check .` and `uv run --extra dev ruff format --check .`: check lint and formatting.

Preserve `no-build = true` and `package = false`: require compatible wheels and run local code without building. `uv pip` also requires wheels. Never bypass these settings. Intel macOS 13 pins OpenCV 4.12.0.88, NumPy 2.2.6, and PySide6 Essentials 6.11.2; recheck wheels before upgrades. For cache restrictions, set `UV_CACHE_DIR=/tmp/calibration-kit-uv-cache`.

## Coding Style & Naming Conventions

Use four-space indentation, Ruff formatting, and 100-character lines. Use `snake_case` for modules/functions and `PascalCase` for classes; preserve mathematical names (`K1`, `R`, `T`). Keep geometry independent of GUI/hardware. Include measurement units (`_m`, `_px`).

## Testing Guidelines

Use pytest with `test_*.py` files and descriptive `test_*` functions. Cover changed numerical behavior, schema validation, and relevant Qt interactions. Use `tmp_path` for generated sessions. GUI tests default to offscreen Qt; explicitly run `QT_QPA_PLATFORM=offscreen uv run --extra dev pytest -q` on headless hosts. No coverage threshold is configured. Hardware changes also require validation with actual cameras; synthetic success does not establish physical accuracy.

## Commit & Pull Request Guidelines

No Git metadata is available to verify historical conventions. Use imperative commit subjects, such as `Fix stereo point validation`. PRs should describe changes, report validation, link issues, and include GUI screenshots. Update docs for workflow or validation changes.

## Calibration & Configuration Safeguards

Preserve metre units and `X_right = R @ X_left + T`. Retain the synthetic-to-live safeguard. Install only the configured OpenCV package to avoid competing `cv2` providers. Host grab timestamps do not prove exposure synchronization.
