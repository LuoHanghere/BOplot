from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_env_config, resolve_workspace, write_json


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _run_tui(session: Any, command: str) -> None:
    if hasattr(session, "tui") and hasattr(session.tui, "execute_command"):
        session.tui.execute_command(command)
        return
    if hasattr(session, "execute_tui_command"):
        session.execute_tui_command(command)
        return
    raise RuntimeError("tui command not supported")


def _resolve_fluent_exe(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_file():
        return path
    return path / "ntbin" / "win64" / "fluent.exe"


def _journal_commands(cad_path: Path, mesh_path: Path, mesh_size: float, stage: str) -> list[str]:
    if stage == "smoke":
        return ["exit yes"]
    if stage == "read_cad":
        return [
            '/file/read-cad "{}"'.format(str(cad_path)),
            "exit yes",
        ]
    if stage == "write_mesh":
        return [
            '/file/read-cad "{}"'.format(str(cad_path)),
            '/file/write-mesh "{}"'.format(str(mesh_path)),
            "exit yes",
        ]
    return [
        '/file/read-cad "{}"'.format(str(cad_path)),
        "/mesh/auto-mesh {} {} {}".format(mesh_size, mesh_size, mesh_size),
        '/file/write-mesh "{}"'.format(str(mesh_path)),
        "exit yes",
    ]


def _run_command(cmd: list[str], timeout_seconds: int) -> tuple[int, str, str, bool]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        return proc.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except Exception:
            stdout = ""
            stderr = ""
        return 124, stdout, stderr, True


def _run_meshing_journal(cad_path: Path, mesh_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    fluent_root = str(config.get("fluent_path", ""))
    if not fluent_root:
        fluent_root = r"C:\Program Files\ANSYS Inc\v231\fluent"
    fluent_exe = _resolve_fluent_exe(fluent_root)
    if not fluent_exe.exists():
        return {"success": False, "message": "fluent exe not found: {}".format(fluent_exe)}
    logs_dir = mesh_path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    mesh_size = float(_safe_float(config.get("mesh_size"), 0.05))
    timeout_seconds = int(_safe_float(config.get("timeout_seconds"), 600))
    show_gui = bool(config.get("show_gui", False))
    stage = str(config.get("stage", "full")).strip().lower()
    journal_path = logs_dir / "meshing_{}.jou".format(stage)
    run_log_path = logs_dir / "meshing_{}.log".format(stage)
    result_path = logs_dir / "meshing_{}_result.json".format(stage)
    journal_path.write_text(
        "\n".join(_journal_commands(cad_path, mesh_path, mesh_size, stage)),
        encoding="utf-8",
    )
    cmd = [
        str(fluent_exe),
        "2d",
        "-meshing",
        "-i",
        str(journal_path),
    ]
    if not show_gui:
        cmd.insert(2, "-g")
    returncode, stdout, stderr, timeout = _run_command(cmd, timeout_seconds)
    run_log_path.write_text(
        "CMD: {}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}\n".format(" ".join(cmd), stdout, stderr),
        encoding="utf-8",
    )
    if timeout:
        return {
            "success": False,
            "message": "meshing timeout after {}s".format(timeout_seconds),
            "journal": str(journal_path),
            "run_log": str(run_log_path),
            "result_file": str(result_path),
            "stage": stage,
        }
    transcript = ""
    match = re.search(r'Opening input/output transcript to file "([^"]+)"', stdout)
    if match:
        transcript = match.group(1)
    if "Mesher mode is not supported - Starting Solver mode" in stdout:
        return {
            "success": False,
            "message": "mesher mode is not supported by current Fluent license",
            "returncode": returncode,
            "journal": str(journal_path),
            "run_log": str(run_log_path),
            "result_file": str(result_path),
            "transcript": transcript,
            "stage": stage,
        }
    return {
        "success": returncode == 0,
        "message": "ok" if returncode == 0 else stderr.strip() or stdout.strip(),
        "returncode": returncode,
        "journal": str(journal_path),
        "run_log": str(run_log_path),
        "result_file": str(result_path),
        "transcript": transcript,
        "stage": stage,
    }


def _run_meshing(cad_path: Path, mesh_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    import ansys.fluent.core as pyfluent

    processors = int(_safe_float(config.get("processors"), 1))
    mesh_size = float(_safe_float(config.get("mesh_size"), 0.05))
    launch_kwargs = {
        "precision": "double",
        "processor_count": processors,
        "mode": "meshing",
        "show_gui": False,
    }
    fluent_path = config.get("fluent_path")
    if fluent_path:
        launch_kwargs["fluent_path"] = str(fluent_path)
    start = time.perf_counter()
    try:
        session = pyfluent.launch_fluent(**launch_kwargs)
    except TypeError:
        launch_kwargs.pop("fluent_path", None)
        session = pyfluent.launch_fluent(**launch_kwargs)
    try:
        try:
            _run_tui(session, '/file/read-cad "{}"'.format(str(cad_path)))
            _run_tui(session, "/mesh/auto-mesh {} {} {}".format(mesh_size, mesh_size, mesh_size))
            _run_tui(session, '/file/write-mesh "{}"'.format(str(mesh_path)))
            cost = time.perf_counter() - start
            return {"success": True, "message": "ok", "cost_seconds": cost}
        except RuntimeError as exc:
            if "tui command not supported" in str(exc):
                return _run_meshing_journal(cad_path, mesh_path, config)
            raise
    finally:
        session.exit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--cad", required=True)
    parser.add_argument("--mesh", default="mesh_from_cad.msh")
    parser.add_argument("--processors", default="1")
    parser.add_argument("--mesh-size", default="0.05")
    parser.add_argument("--use-journal", default="1")
    parser.add_argument("--show-gui", default="0")
    parser.add_argument("--timeout-seconds", default="600")
    parser.add_argument("--stage", default="full")
    args = parser.parse_args()
    data_dir = SCRIPT_DIR / "data"
    workspace = resolve_workspace(args.workspace, base_dir=data_dir)
    cad_path = Path(args.cad)
    if not cad_path.is_absolute():
        cad_path = workspace / args.cad
    mesh_path = Path(args.mesh)
    if not mesh_path.is_absolute():
        mesh_path = workspace / args.mesh
    env_cfg = read_env_config(SCRIPT_DIR.parent)
    fluent_path = env_cfg.get("fluent")
    if not fluent_path:
        fluent_path = r"C:\Program Files\ANSYS Inc\v231\fluent"
    config = {
        "processors": int(args.processors),
        "mesh_size": float(args.mesh_size),
        "fluent_path": fluent_path,
        "show_gui": str(args.show_gui).strip() in {"1", "true", "True"},
        "timeout_seconds": int(args.timeout_seconds),
        "stage": str(args.stage).strip().lower(),
    }
    try:
        if str(args.use_journal).strip() in {"1", "true", "True"}:
            payload = _run_meshing_journal(cad_path, mesh_path, config)
        else:
            payload = _run_meshing(cad_path, mesh_path, config)
    except Exception as exc:
        payload = {"success": False, "message": "meshing failed: {}".format(exc)}
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stage = str(config.get("stage", "full"))
    staged_result = logs_dir / "meshing_{}_result.json".format(stage)
    write_json(staged_result, payload)
    write_json(workspace / "meshing_result.json", payload)
    print(str(mesh_path))


if __name__ == "__main__":
    main()
