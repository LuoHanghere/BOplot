from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plat_bo.optimizer.ui_app import run_ui  # noqa: E402


if __name__ == "__main__":
    run_ui()

