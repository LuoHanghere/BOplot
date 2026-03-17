from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ensure_parameters, read_json, resolve_workspace, write_json


def _default_x_positions() -> list[float]:
    return [0.1, 0.3, 0.5, 0.7, 0.9]


def build_geometry(input_path: Path, workspace: Path) -> Path:
    data = read_json(input_path)
    parameters = ensure_parameters(data, count=5)
    x_positions = data.get("x_positions")
    if not isinstance(x_positions, list) or len(x_positions) != 5:
        x_positions = _default_x_positions()
    x_positions = [float(x) for x in x_positions]
    upper = [{"x": x, "y": float(y)} for x, y in zip(x_positions, parameters)]
    lower = [{"x": x, "y": -float(y)} for x, y in zip(x_positions, parameters)]
    farfield = data.get("farfield")
    if not isinstance(farfield, dict):
        farfield = {"xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0}
    geometry = {"upper": upper, "lower": lower, "farfield": farfield}
    output_path = workspace / "geometry.json"
    write_json(output_path, geometry)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    data_dir = SCRIPT_DIR / "data"
    workspace = resolve_workspace(args.workspace, base_dir=data_dir)
    geometry_path = build_geometry(Path(args.input), workspace)
    print(str(geometry_path))


if __name__ == "__main__":
    main()
