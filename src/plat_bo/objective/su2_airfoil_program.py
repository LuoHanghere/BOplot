from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_text(value: object, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_pipeline(repo_root: Path, pipeline: Path, input_path: Path, workspace_name: str, problem_cfg: dict[str, Any], mesh_level: str) -> tuple[bool, dict[str, Any]]:
    cmd = [
        sys.executable,
        str(pipeline),
        "--input",
        str(input_path),
        "--workspace",
        workspace_name,
        "--mesh-level",
        mesh_level,
        "--iterations",
        str(_safe_int(problem_cfg.get("iterations"), 300)),
        "--timeout-seconds",
        str(_safe_int(problem_cfg.get("timeout_seconds"), 1800)),
        "--solver-mode",
        _safe_text(problem_cfg.get("solver_mode"), "su2"),
        "--su2-cfd",
        _safe_text(problem_cfg.get("su2_cfd"), ""),
        "--processors",
        str(_safe_int(problem_cfg.get("processors"), 4)),
        "--mpi-exec",
        _safe_text(problem_cfg.get("mpi_exec"), ""),
        "--allow-mock-fallback",
        "1" if bool(problem_cfg.get("allow_mock_fallback", False)) else "0",
    ]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, {
            "success": False,
            "message": proc.stderr.strip() or proc.stdout.strip() or "su2 pipeline failed",
            "command": cmd,
        }
    return True, {"success": True, "command": cmd}


def _read_result(su2_dir: Path, workspace_name: str) -> tuple[bool, dict[str, Any]]:
    result_path = su2_dir / "data" / workspace_name / "result.json"
    if not result_path.exists():
        return False, {
            "success": False,
            "message": "su2 result missing: {}".format(result_path),
        }
    result = _read_json(result_path)
    if not bool(result.get("success")):
        return False, {
            "success": False,
            "message": str(result.get("message", "su2 solve failed")),
            "workspace": workspace_name,
        }
    return True, result


def _as_output(
    result: dict[str, Any],
    workspace_name: str,
    mesh_level_used: str,
    primary_workspace: str,
    review_workspace: str | None,
    review_mesh_level: str | None,
    objective_drag_floor: float,
) -> dict[str, Any]:
    cl = float(result.get("lift_coefficient"))
    cd = float(result.get("drag_coefficient"))
    cd_for_objective = max(cd, objective_drag_floor)
    objective = -(cl / cd_for_objective)
    objective_vector_raw = result.get("objective_vector")
    if isinstance(objective_vector_raw, list) and len(objective_vector_raw) >= 2:
        objective_vector = [float(objective_vector_raw[0]), float(objective_vector_raw[1])]
    else:
        objective_vector = [-cl, cd]
    payload = {
        "objective": objective,
        "objective_vector": objective_vector,
        "lift_coefficient": cl,
        "drag_coefficient": cd,
        "drag_for_objective": cd_for_objective,
        "objective_drag_floor": objective_drag_floor,
        "success": True,
        "message": "ok",
        "workspace": workspace_name,
        "mesh_level_used": mesh_level_used,
        "primary_workspace": primary_workspace,
        "review_workspace": review_workspace,
        "review_mesh_level": review_mesh_level,
        "low_workspace": primary_workspace,
        "medium_workspace": review_workspace,
    }
    return payload


def run(input_path: Path, output_path: Path) -> None:
    payload = _read_json(input_path)
    params = payload.get("parameters", [])
    if not isinstance(params, list) or len(params) != 5:
        _write_json(
            output_path,
            {
                "objective": float("nan"),
                "success": False,
                "message": "su2_airfoil_2d needs 5 parameters",
            },
        )
        return

    repo_root = Path(__file__).resolve().parents[3]
    su2_dir = repo_root / "SU2"
    pipeline = su2_dir / "run_cad_su2_pipeline.py"
    if not pipeline.exists():
        _write_json(
            output_path,
            {
                "objective": float("nan"),
                "success": False,
                "message": "SU2 pipeline script missing: {}".format(pipeline),
            },
        )
        return

    problem_cfg = payload.get("problem_config", {})
    if not isinstance(problem_cfg, dict):
        problem_cfg = {}
    task_id = _safe_text(payload.get("task_id"), "su2-task")
    iteration = _safe_int(payload.get("iteration"), 0)
    objective_drag_floor = max(1e-4, float(problem_cfg.get("objective_drag_floor", 0.005)))
    medium_review_cd_threshold = max(objective_drag_floor, float(problem_cfg.get("medium_review_cd_threshold", objective_drag_floor)))
    project_name = _safe_text(problem_cfg.get("project_name"), task_id)
    iter_tag = "{}_iter_{:04d}".format(task_id, max(0, iteration))
    base_mesh_level = _safe_text(problem_cfg.get("mesh_level"), "low").lower()
    review_mesh_level = _safe_text(problem_cfg.get("review_mesh_level"), "medium").lower()
    if review_mesh_level == base_mesh_level:
        review_mesh_level = "high" if base_mesh_level == "medium" else "medium"
    primary_workspace = _safe_text(
        problem_cfg.get("workspace_primary"),
        str(Path(project_name) / "{}_{}".format(iter_tag, base_mesh_level)),
    )
    review_workspace = _safe_text(
        problem_cfg.get("workspace_review"),
        str(Path(project_name) / "{}_{}".format(iter_tag, review_mesh_level)),
    )
    trial_input_path = su2_dir / "data" / project_name / "{}_trial_input.json".format(iter_tag)
    _write_json(
        trial_input_path,
        {
            "task_id": task_id,
            "iteration": iteration,
            "parameters": [float(v) for v in params],
            "x_positions": problem_cfg.get("x_positions"),
            "shape_scale": problem_cfg.get("shape_scale", 0.2),
            "farfield": problem_cfg.get("farfield"),
            "su2": problem_cfg.get("su2", {}),
        },
    )
    ok_primary, primary_meta = _run_pipeline(
        repo_root=repo_root,
        pipeline=pipeline,
        input_path=trial_input_path,
        workspace_name=primary_workspace,
        problem_cfg=problem_cfg,
        mesh_level=base_mesh_level,
    )
    if not ok_primary:
        _write_json(
            output_path,
            {
                "objective": float("nan"),
                "success": False,
                "message": str(primary_meta.get("message", "su2 primary-level run failed")),
                "workspace": primary_workspace,
            },
        )
        return
    ok_primary_result, primary_result = _read_result(su2_dir, primary_workspace)
    if not ok_primary_result:
        _write_json(
            output_path,
            {
                "objective": float("nan"),
                "success": False,
                "message": str(primary_result.get("message", "su2 primary-level result failed")),
                "workspace": primary_workspace,
            },
        )
        return

    review_enabled = bool(problem_cfg.get("enable_medium_review", True))
    if not review_enabled:
        _write_json(
            output_path,
            _as_output(
                primary_result,
                primary_workspace,
                base_mesh_level,
                primary_workspace,
                None,
                None,
                objective_drag_floor,
            ),
        )
        return

    state_path = su2_dir / "data" / project_name / "review_state.json"
    state_data = _read_json(state_path) if state_path.exists() else {}
    best_low = state_data.get("best_low_objective")
    primary_objective = float(primary_result.get("objective"))
    primary_cd = float(primary_result.get("drag_coefficient"))
    should_review = best_low is None or primary_objective < float(best_low) or primary_cd < medium_review_cd_threshold
    if should_review:
        ok_review, review_meta = _run_pipeline(
            repo_root=repo_root,
            pipeline=pipeline,
            input_path=trial_input_path,
            workspace_name=review_workspace,
            problem_cfg=problem_cfg,
            mesh_level=review_mesh_level,
        )
        if not ok_review:
            _write_json(
                output_path,
                {
                    "objective": float("nan"),
                    "success": False,
                    "message": str(review_meta.get("message", "su2 review-level run failed")),
                    "workspace": review_workspace,
                },
            )
            return
        ok_review_result, review_result = _read_result(su2_dir, review_workspace)
        if not ok_review_result:
            _write_json(
                output_path,
                {
                    "objective": float("nan"),
                    "success": False,
                    "message": str(review_result.get("message", "su2 review-level result failed")),
                    "workspace": review_workspace,
                },
            )
            return
        state_data["best_low_objective"] = primary_objective
        _write_json(state_path, state_data)
        _write_json(
            output_path,
            _as_output(
                review_result,
                review_workspace,
                review_mesh_level,
                primary_workspace,
                review_workspace,
                review_mesh_level,
                objective_drag_floor,
            ),
        )
        return

    _write_json(
        output_path,
        _as_output(
            primary_result,
            primary_workspace,
            base_mesh_level,
            primary_workspace,
            None,
            None,
            objective_drag_floor,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
