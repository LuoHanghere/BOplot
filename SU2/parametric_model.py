from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ensure_parameters, read_json, resolve_workspace, write_json


def _default_x_positions() -> list[float]:
    return [0.1, 0.3, 0.5, 0.7, 0.9]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _base_upper() -> list[float]:
    return [0.03, 0.065, 0.07, 0.045, 0.018]


def _base_lower() -> list[float]:
    return [-0.015, -0.022, -0.02, -0.012, -0.006]


def _resolve_x_positions(raw: object) -> list[float]:
    default = _default_x_positions()
    if not isinstance(raw, list) or len(raw) != 5:
        return default
    values = [float(v) for v in raw]
    clipped = [_clamp(v, 0.05, 0.95) for v in values]
    clipped.sort()
    return clipped


def build_geometry(input_path: Path, workspace: Path) -> Path:
    data = read_json(input_path)
    parameters = ensure_parameters(data, count=5)
    x_positions = _resolve_x_positions(data.get("x_positions"))
    scale = float(data.get("shape_scale", 0.2))
    upper_base = _base_upper()
    lower_base = _base_lower()
    upper: list[dict[str, float]] = []
    lower: list[dict[str, float]] = []
    min_gap = 0.012
    for x, p, up_base, low_base in zip(x_positions, parameters, upper_base, lower_base):
        delta = scale * float(p)
        up = _clamp(up_base + delta, -0.01, 0.16)
        low = _clamp(low_base + 0.45 * delta, -0.12, 0.03)
        if up - low < min_gap:
            low = up - min_gap
        upper.append({"x": x, "y": up})
        lower.append({"x": x, "y": low})
    farfield = data.get("farfield")
    if not isinstance(farfield, dict):
        farfield = {"xmin": -8.0, "xmax": 12.0, "ymin": -8.0, "ymax": 8.0}
    geometry = {
        "upper": upper,
        "lower": lower,
        "farfield": farfield,
        "constraints": {
            "min_gap": min_gap,
            "shape_scale": scale,
            "base_upper": upper_base,
            "base_lower": lower_base,
        },
    }
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
