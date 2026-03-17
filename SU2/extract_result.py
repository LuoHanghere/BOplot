from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_json, resolve_workspace, write_json


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_result(workspace: Path, source: str = "su2_solver_output.json", output: str = "result.json") -> Path:
    source_path = workspace / source
    if not source_path.exists():
        raise FileNotFoundError("solver output missing: {}".format(source_path))
    payload = read_json(source_path)
    success = bool(payload.get("success"))
    cl = _to_float(payload.get("lift_coefficient"))
    cd = _to_float(payload.get("drag_coefficient"))
    if success:
        objective = _to_float(payload.get("objective"), -(cl / max(cd, 1e-6)))
        objective_vector_raw = payload.get("objective_vector")
        if isinstance(objective_vector_raw, list) and len(objective_vector_raw) >= 2:
            objective_vector = [_to_float(objective_vector_raw[0]), _to_float(objective_vector_raw[1])]
        else:
            objective_vector = [-cl, cd]
        result = {
            "objective": objective,
            "objective_vector": objective_vector,
            "lift_coefficient": cl,
            "drag_coefficient": cd,
            "success": True,
            "message": str(payload.get("message", "ok")),
            "mode": str(payload.get("mode", "")),
            "cost_seconds": _to_float(payload.get("cost_seconds"), 0.0),
        }
    else:
        result = {
            "objective": float("nan"),
            "objective_vector": None,
            "lift_coefficient": cl,
            "drag_coefficient": cd,
            "success": False,
            "message": str(payload.get("message", "su2 solve failed")),
            "mode": str(payload.get("mode", "")),
            "cost_seconds": _to_float(payload.get("cost_seconds"), 0.0),
        }
    output_path = workspace / output
    write_json(output_path, result)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--source", default="su2_solver_output.json")
    parser.add_argument("--output", default="result.json")
    args = parser.parse_args()
    workspace = resolve_workspace(args.workspace, base_dir=SCRIPT_DIR / "data")
    result_path = extract_result(workspace, source=str(args.source), output=str(args.output))
    print(str(result_path))


if __name__ == "__main__":
    main()
