"""Run the application from source without building or installing a package."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    runpy.run_module("calibration_demo.cli", run_name="__main__")
