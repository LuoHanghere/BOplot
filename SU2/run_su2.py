from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_env_config, read_json, resolve_workspace, write_json


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mock_result(workspace: Path) -> dict[str, Any]:
    geometry_path = workspace / "geometry.json"
    if geometry_path.exists():
        geometry = read_json(geometry_path)
        upper = geometry.get("upper", [])
        ys = [abs(float(p.get("y", 0.0))) for p in upper if isinstance(p, dict)]
    else:
        ys = []
    camber = sum(ys) / max(1, len(ys))
    cl = 0.72 + 2.4 * camber
    cd = 0.018 + 0.55 * camber * camber
    return {
        "lift_coefficient": cl,
        "drag_coefficient": cd,
        "objective": -(cl / max(cd, 1e-6)),
        "objective_vector": [-cl, cd],
        "success": True,
        "message": "mock",
    }


def _resolve_su2_cfd(configured: str, env_cfg: dict[str, str]) -> Path | None:
    raw = configured.strip()
    if raw:
        path = Path(raw)
        if path.is_dir():
            exe = path / "SU2_CFD.exe"
            if exe.exists():
                return exe
        if path.exists():
            return path
    env_candidate = str(env_cfg.get("su2_cfd", "")).strip()
    if env_candidate:
        path = Path(env_candidate)
        if path.is_dir():
            exe = path / "SU2_CFD.exe"
            if exe.exists():
                return exe
        if path.exists():
            return path
    su2_root = str(env_cfg.get("su2_root", "")).strip()
    if su2_root:
        exe = Path(su2_root) / "bin" / "SU2_CFD.exe"
        if exe.exists():
            return exe
    which_hit = shutil.which("SU2_CFD")
    if which_hit:
        return Path(which_hit)
    which_hit_exe = shutil.which("SU2_CFD.exe")
    if which_hit_exe:
        return Path(which_hit_exe)
    return None


def _resolve_mpi_exec(configured: str, env_cfg: dict[str, str]) -> Path | None:
    raw = configured.strip()
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    env_candidate = str(env_cfg.get("mpi_exec", "")).strip()
    if env_candidate:
        path = Path(env_candidate)
        if path.exists():
            return path
    which_hit = shutil.which("mpiexec")
    if which_hit:
        return Path(which_hit)
    which_hit_exe = shutil.which("mpiexec.exe")
    if which_hit_exe:
        return Path(which_hit_exe)
    return None


def _find_history_file(workspace: Path) -> Path | None:
    candidates = [
        workspace / "history.csv",
        workspace / "history.dat",
        workspace / "history",
    ]
    for item in candidates:
        if item.exists() and item.is_file():
            return item
    csv_like = sorted(workspace.glob("history*.csv"))
    if csv_like:
        return csv_like[-1]
    return None


def _pick_value(row: dict[str, str], keywords: list[str]) -> float | None:
    lowered = {k.strip().lower(): v for k, v in row.items() if isinstance(k, str)}
    for key, value in lowered.items():
        if any(word in key for word in keywords):
            parsed = _safe_float(value, float("nan"))
            if parsed == parsed:
                return parsed
    return None


def _parse_history(history_path: Path) -> tuple[float, float]:
    if history_path.suffix.lower() == ".csv":
        with history_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            last_row: dict[str, str] | None = None
            for row in reader:
                if row:
                    last_row = row
            if not last_row:
                raise ValueError("history.csv has no data rows")
            cl = _pick_value(last_row, ["lift", "cl"])
            cd = _pick_value(last_row, ["drag", "cd"])
            if cl is None or cd is None:
                raise ValueError("cannot parse lift/drag columns from {}".format(history_path))
            return cl, cd

    text = history_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("history file is empty")
    numeric = [token for token in lines[-1].replace(",", " ").split(" ") if token]
    floats: list[float] = []
    for token in numeric:
        value = _safe_float(token, float("nan"))
        if value == value:
            floats.append(value)
    if len(floats) < 2:
        raise ValueError("cannot parse numeric data in history line")
    return floats[-2], floats[-1]


def run_solver(
    workspace: Path,
    cfg: str,
    mode: str,
    timeout_seconds: int,
    su2_cfd: str,
    processors: int,
    mpi_exec: str,
) -> Path:
    result_path = workspace / "su2_solver_output.json"
    if mode == "mock":
        payload = _mock_result(workspace)
        payload["mode"] = "mock"
        payload["cost_seconds"] = 0.0
        write_json(result_path, payload)
        return result_path

    env_cfg = read_env_config(SCRIPT_DIR.parent)
    su2_exe = _resolve_su2_cfd(su2_cfd, env_cfg)
    if su2_exe is None:
        write_json(
            result_path,
            {
                "success": False,
                "mode": "su2",
                "message": "SU2_CFD executable not found",
            },
        )
        return result_path

    start = time.perf_counter()
    mpi_path = _resolve_mpi_exec(mpi_exec, env_cfg) if processors > 1 else None
    if processors > 1 and mpi_path is None:
        write_json(
            result_path,
            {
                "success": False,
                "mode": "su2",
                "message": "mpiexec not found for processors={}".format(processors),
            },
        )
        return result_path

    cmd = [str(su2_exe), cfg] if processors <= 1 else [str(mpi_path), "-n", str(processors), str(su2_exe), cfg]
    try:
        completed = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        write_json(
            result_path,
            {
                "success": False,
                "mode": "su2",
                "message": "timeout after {}s".format(timeout_seconds),
                "command": cmd,
            },
        )
        return result_path

    (workspace / "su2_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (workspace / "su2_stderr.log").write_text(completed.stderr, encoding="utf-8")

    if completed.returncode != 0:
        write_json(
            result_path,
            {
                "success": False,
                "mode": "su2",
                "message": completed.stderr.strip() or completed.stdout.strip() or "su2 failed",
                "command": cmd,
                "returncode": completed.returncode,
            },
        )
        return result_path

    history_path = _find_history_file(workspace)
    if history_path is None:
        write_json(
            result_path,
            {
                "success": False,
                "mode": "su2",
                "message": "history file not found after SU2 run",
                "command": cmd,
                "returncode": completed.returncode,
            },
        )
        return result_path

    try:
        cl, cd = _parse_history(history_path)
    except Exception as exc:
        write_json(
            result_path,
            {
                "success": False,
                "mode": "su2",
                "message": "parse history failed: {}".format(exc),
                "history": str(history_path),
            },
        )
        return result_path

    cd_abs = abs(cd)
    payload = {
        "lift_coefficient": cl,
        "drag_coefficient": cd_abs,
        "drag_coefficient_raw": cd,
        "objective": -(cl / max(cd_abs, 1e-6)),
        "objective_vector": [-cl, cd_abs],
        "success": True,
        "message": "ok",
        "mode": "su2",
        "command": cmd,
        "returncode": completed.returncode,
        "history": str(history_path),
        "cost_seconds": time.perf_counter() - start,
    }
    write_json(result_path, payload)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--cfg", default="case.cfg")
    parser.add_argument("--mode", default="su2")
    parser.add_argument("--timeout-seconds", default="1800")
    parser.add_argument("--su2-cfd", default="")
    parser.add_argument("--processors", default="1")
    parser.add_argument("--mpi-exec", default="")
    args = parser.parse_args()
    workspace = resolve_workspace(args.workspace, base_dir=SCRIPT_DIR / "data")
    processors = int(str(args.processors).strip() or "1")
    output_path = run_solver(
        workspace=workspace,
        cfg=str(args.cfg),
        mode=str(args.mode).strip().lower(),
        timeout_seconds=int(args.timeout_seconds),
        su2_cfd=str(args.su2_cfd),
        processors=processors,
        mpi_exec=str(args.mpi_exec),
    )
    print(str(output_path))


if __name__ == "__main__":
    main()
