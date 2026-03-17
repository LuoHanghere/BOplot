from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_json, resolve_workspace, write_json


def _run(command: list[str], log_path: Path, timeout_seconds: int) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
        log_path.write_text(
            "CMD: {}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}\n".format(" ".join(command), result.stdout, result.stderr),
            encoding="utf-8",
        )
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "log": str(log_path),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        log_path.write_text(
            "CMD: {}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}\n".format(" ".join(command), stdout, stderr),
            encoding="utf-8",
        )
        return {
            "success": False,
            "returncode": -1,
            "log": str(log_path),
            "message": "timeout after {}s".format(timeout_seconds),
        }


def _validate_stage(name: str, workspace: Path, stage_result: dict[str, Any]) -> dict[str, Any]:
    if not stage_result.get("success"):
        return stage_result
    if name == "gmsh_mesh":
        mesh_path = workspace / "mesh.msh"
        if not mesh_path.exists():
            stage_result["success"] = False
            stage_result["message"] = "mesh file missing: {}".format(mesh_path)
    if name == "fluent_solve":
        result_path = workspace / "result.json"
        if not result_path.exists():
            stage_result["success"] = False
            stage_result["message"] = "solver result missing: {}".format(result_path)
        else:
            payload = read_json(result_path)
            if not bool(payload.get("success")):
                stage_result["success"] = False
                stage_result["message"] = str(payload.get("message", "fluent solve failed"))
    return stage_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--mesh-size", default="0.05")
    parser.add_argument("--iterations", default="150")
    parser.add_argument("--timeout-seconds", default="600")
    parser.add_argument("--run-cad-probe", default="0")
    parser.add_argument("--solver-mode", default="fluent")
    parser.add_argument("--allow-mock-fallback", default="1")
    args = parser.parse_args()
    workspace = resolve_workspace(args.workspace, base_dir=SCRIPT_DIR / "data")
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timeout_seconds = int(args.timeout_seconds)
    run_cad_probe = str(args.run_cad_probe).strip() in {"1", "true", "True"}
    allow_mock_fallback = str(args.allow_mock_fallback).strip() in {"1", "true", "True"}
    solver_mode = str(args.solver_mode).strip().lower()
    py = sys.executable

    steps: list[tuple[str, list[str], bool]] = [
        (
            "geometry",
            [
                py,
                str(SCRIPT_DIR / "parametric_model.py"),
                "--input",
                str(Path(args.input)),
                "--workspace",
                str(args.workspace),
            ],
            True,
        ),
        (
            "gmsh_cad",
            [
                py,
                str(SCRIPT_DIR / "export_cad.py"),
                "--workspace",
                str(args.workspace),
                "--geometry",
                "geometry.json",
                "--format",
                "step",
            ],
            True,
        ),
        (
            "spaceclaim_export",
            [
                py,
                str(SCRIPT_DIR / "spaceclaim_mesh.py"),
                "--workspace",
                str(args.workspace),
                "--input",
                "airfoil.step",
                "--export-format",
                "step",
                "--timeout-seconds",
                str(args.timeout_seconds),
            ],
            True,
        ),
        (
            "gmsh_mesh",
            [
                py,
                str(SCRIPT_DIR / "mesh.py"),
                "--workspace",
                str(args.workspace),
                "--geometry",
                "geometry.json",
                "--mesh-size",
                str(args.mesh_size),
                "--msh-version",
                "2.2",
                "--mesh-binary",
                "0",
            ],
            True,
        ),
        (
            "fluent_solve",
            [
                py,
                str(SCRIPT_DIR / "run_fluent.py"),
                "--workspace",
                str(args.workspace),
                "--mesh",
                "mesh.msh",
                "--mode",
                solver_mode,
                "--iterations",
                str(args.iterations),
                "--processors",
                "1",
            ],
            True,
        ),
    ]
    if run_cad_probe:
        steps.insert(
            3,
            (
                "fluent_read_cad_probe",
                [
                    py,
                    str(SCRIPT_DIR / "fluent_meshing.py"),
                    "--workspace",
                    str(args.workspace),
                    "--cad",
                    "airfoil_from_spaceclaim.step",
                    "--stage",
                    "read_cad",
                    "--timeout-seconds",
                    "120",
                    "--use-journal",
                    "1",
                ],
                False,
            ),
        )

    results: dict[str, Any] = {}
    ok = True
    used_mock_fallback = False
    for name, command, required in steps:
        stage_log = logs_dir / "{}.log".format(name)
        stage_result = _run(command, stage_log, timeout_seconds)
        stage_result = _validate_stage(name, workspace, stage_result)
        results[name] = stage_result
        if (
            name == "fluent_solve"
            and not stage_result.get("success")
            and allow_mock_fallback
            and solver_mode != "mock"
        ):
            fallback_command = [
                py,
                str(SCRIPT_DIR / "run_fluent.py"),
                "--workspace",
                str(args.workspace),
                "--mesh",
                "mesh.msh",
                "--mode",
                "mock",
                "--iterations",
                str(args.iterations),
                "--processors",
                "1",
            ]
            fallback_log = logs_dir / "fluent_solve_mock.log"
            fallback_result = _run(fallback_command, fallback_log, timeout_seconds)
            fallback_result = _validate_stage("fluent_solve", workspace, fallback_result)
            results["fluent_solve_mock"] = fallback_result
            if fallback_result.get("success"):
                used_mock_fallback = True
                continue
        if required and not stage_result.get("success"):
            ok = False
            break

    payload = {
        "success": ok,
        "workspace": str(workspace),
        "stages": results,
        "used_mock_fallback": used_mock_fallback,
    }
    write_json(logs_dir / "pipeline_result.json", payload)
    write_json(workspace / "pipeline_result.json", payload)
    print(str(logs_dir / "pipeline_result.json"))


if __name__ == "__main__":
    main()
