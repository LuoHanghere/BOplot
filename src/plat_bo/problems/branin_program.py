from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def branin(x1: float, x2: float) -> float:
    a = 1.0
    b = 5.1 / (4.0 * math.pi * math.pi)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1.0 - t) * math.cos(x1) + s


def run(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    params = payload.get("parameters", [])
    if not isinstance(params, list) or len(params) != 2:
        out = {"objective": float("nan"), "success": False, "message": "branin needs 2 parameters"}
    else:
        x1 = float(params[0])
        x2 = float(params[1])
        value = branin(x1, x2)
        out = {"objective": value, "success": True, "message": "ok"}

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Branin evaluator program.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()

