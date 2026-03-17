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
        su2_mesh = workspace / "mesh.su2"
        vtk_mesh = workspace / "mesh.vtk"
        if not su2_mesh.exists():
            stage_result["success"] = False
            stage_result["message"] = "mesh file missing: {}".format(su2_mesh)
        elif not vtk_mesh.exists():
            stage_result["success"] = False
            stage_result["message"] = "vtk mesh missing: {}".format(vtk_mesh)
    if name == "su2_solve":
        result_path = workspace / "su2_solver_output.json"
        if not result_path.exists():
            stage_result["success"] = False
            stage_result["message"] = "solver result missing: {}".format(result_path)
        else:
            payload = read_json(result_path)
            if not bool(payload.get("success")):
                stage_result["success"] = False
                stage_result["message"] = str(payload.get("message", "su2 solve failed"))
    if name == "extract_result":
        result_path = workspace / "result.json"
        if not result_path.exists():
            stage_result["success"] = False
            stage_result["message"] = "extract result missing: {}".format(result_path)
    return stage_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--mesh-level", default="medium")
    parser.add_argument("--mesh-size", default="")
    parser.add_argument("--iterations", default="800")
    parser.add_argument("--timeout-seconds", default="1800")
    parser.add_argument("--solver-mode", default="su2")
    parser.add_argument("--su2-cfd", default="")
    parser.add_argument("--processors", default="1")
    parser.add_argument("--mpi-exec", default="")
    parser.add_argument("--allow-mock-fallback", default="1")
    args = parser.parse_args()
    workspace = resolve_workspace(args.workspace, base_dir=SCRIPT_DIR / "data")
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timeout_seconds = int(args.timeout_seconds)
    allow_mock_fallback = str(args.allow_mock_fallback).strip() in {"1", "true", "True"}
    solver_mode = str(args.solver_mode).strip().lower()
    processors = int(str(args.processors).strip() or "1")
    py = sys.executable

    mesh_command = [
        py,
        str(SCRIPT_DIR / "mesh.py"),
        "--workspace",
        str(args.workspace),
        "--geometry",
        "geometry.json",
        "--mesh-level",
        str(args.mesh_level),
        "--msh-version",
        "2.2",
        "--mesh-binary",
        "0",
    ]
    if str(args.mesh_size).strip():
        mesh_command.extend(["--mesh-size", str(args.mesh_size)])

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
        ("gmsh_mesh", mesh_command, True),
        (
            "setup_case",
            [
                py,
                str(SCRIPT_DIR / "setup_case.py"),
                "--workspace",
                str(args.workspace),
                "--input",
                str(Path(args.input)),
                "--mesh",
                "mesh.su2",
                "--iterations",
                str(args.iterations),
            ],
            True,
        ),
        (
            "su2_solve",
            [
                py,
                str(SCRIPT_DIR / "run_su2.py"),
                "--workspace",
                str(args.workspace),
                "--cfg",
                "case.cfg",
                "--mode",
                solver_mode,
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--su2-cfd",
                str(args.su2_cfd),
                "--processors",
                str(processors),
                "--mpi-exec",
                str(args.mpi_exec),
            ],
            True,
        ),
        (
            "extract_result",
            [
                py,
                str(SCRIPT_DIR / "extract_result.py"),
                "--workspace",
                str(args.workspace),
                "--source",
                "su2_solver_output.json",
                "--output",
                "result.json",
            ],
            True,
        ),
        (
            "prepare_visualization",
            [
                py,
                str(SCRIPT_DIR / "prepare_visualization.py"),
                "--workspace",
                str(args.workspace),
            ],
            True,
        ),
    ]

    results: dict[str, Any] = {}
    ok = True
    used_mock_fallback = False
    for name, command, required in steps:
        stage_log = logs_dir / "{}.log".format(name)
        stage_result = _run(command, stage_log, timeout_seconds)
        stage_result = _validate_stage(name, workspace, stage_result)
        results[name] = stage_result
        if name == "su2_solve" and not stage_result.get("success") and allow_mock_fallback and solver_mode != "mock":
            fallback_command = [
                py,
                str(SCRIPT_DIR / "run_su2.py"),
                "--workspace",
                str(args.workspace),
                "--cfg",
                "case.cfg",
                "--mode",
                "mock",
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--processors",
                "1",
            ]
            fallback_log = logs_dir / "su2_solve_mock.log"
            fallback_result = _run(fallback_command, fallback_log, timeout_seconds)
            fallback_result = _validate_stage("su2_solve", workspace, fallback_result)
            results["su2_solve_mock"] = fallback_result
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
        "solver_mode": solver_mode,
        "mesh_level": str(args.mesh_level),
    }
    write_json(logs_dir / "pipeline_result.json", payload)
    write_json(workspace / "pipeline_result.json", payload)
    print(str(logs_dir / "pipeline_result.json"))


if __name__ == "__main__":
    main()
