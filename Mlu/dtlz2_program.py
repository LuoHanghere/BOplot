from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

def dtlz2_3d(x: list[float]) -> list[float]:
    """
    DTLZ2 function with 3 objectives.
    Usually expects x to have length >= 3.
    """
    m = 3
    d = len(x)
    k = d - m + 1
    
    g = sum((x[i] - 0.5) ** 2 for i in range(m - 1, d))
    
    f1 = (1.0 + g) * math.cos(x[0] * math.pi / 2.0) * math.cos(x[1] * math.pi / 2.0)
    f2 = (1.0 + g) * math.cos(x[0] * math.pi / 2.0) * math.sin(x[1] * math.pi / 2.0)
    f3 = (1.0 + g) * math.sin(x[0] * math.pi / 2.0)
    
    return [f1, f2, f3]

def run(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    params = payload.get("parameters", [])
    if not isinstance(params, list) or len(params) < 3:
        out = {
            "objective": float("nan"), 
            "success": False, 
            "message": "dtlz2 needs at least 3 parameters"
        }
    else:
        try:
            x = [float(p) for p in params]
            f_vals = dtlz2_3d(x)
            # In BO, usually we want a single scalar for fallback. 
            # We can return the sum or hypervolume-related proxy as the main objective,
            # but the real MOO logic will use `objective_vector`.
            # We assume minimize for all by default in BO, DTLZ2 is minimization.
            out = {
                "objective": sum(f_vals), # Fallback single objective
                "objective_vector": f_vals, # The real multi-objective values
                "success": True, 
                "message": "ok"
            }
        except Exception as e:
            out = {
                "objective": float("nan"), 
                "success": False, 
                "message": str(e)
            }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone DTLZ2 evaluator program.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.input), Path(args.output))

if __name__ == "__main__":
    main()
