from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_env_config, resolve_workspace, write_json


def _resolve_fluent_exe(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_file():
        return path
    return path / "ntbin" / "win64" / "fluent.exe"


def _launch_cli(workspace: Path, fluent_root: str, timeout_seconds: int, show_gui: bool) -> dict[str, Any]:
    fluent_exe = _resolve_fluent_exe(fluent_root)
    if not fluent_exe.exists():
        return {"success": False, "message": "fluent exe not found: {}".format(fluent_exe)}
    journal_path = workspace / "smoke.jou"
    journal_path.write_text("exit yes\n", encoding="utf-8")
    cmd = [str(fluent_exe), "2d", "-i", str(journal_path)]
    if not show_gui:
        cmd.insert(2, "-g")
    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "cli timeout after {}s".format(timeout_seconds)}
    cost = time.perf_counter() - start
    (workspace / "smoke_cli.log").write_text(
        "CMD: {}\n\nSTDOUT:\n{}\n\nSTDERR:\n{}\n".format(" ".join(cmd), result.stdout, result.stderr),
        encoding="utf-8",
    )
    return {
        "success": result.returncode == 0,
        "message": "ok" if result.returncode == 0 else (result.stderr.strip() or result.stdout.strip()),
        "returncode": result.returncode,
        "cost_seconds": cost,
    }


def _launch_pyfluent(fluent_root: str, timeout_seconds: int, show_gui: bool) -> dict[str, Any]:
    import ansys.fluent.core as pyfluent

    kwargs: dict[str, Any] = {
        "precision": "double",
        "processor_count": 1,
        "mode": "solver",
        "show_gui": show_gui,
    }
    if fluent_root:
        kwargs["fluent_path"] = fluent_root
    start = time.perf_counter()
    try:
        try:
            session = pyfluent.launch_fluent(**kwargs)
        except TypeError:
            kwargs.pop("fluent_path", None)
            session = pyfluent.launch_fluent(**kwargs)
        try:
            _ = timeout_seconds
            cost = time.perf_counter() - start
            return {"success": True, "message": "ok", "cost_seconds": cost}
        finally:
            session.exit()
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--mode", default="cli")
    parser.add_argument("--show-gui", default="0")
    parser.add_argument("--timeout-seconds", default="120")
    args = parser.parse_args()
    data_dir = SCRIPT_DIR / "data"
    workspace = resolve_workspace(args.workspace, base_dir=data_dir)
    show_gui = str(args.show_gui).strip() in {"1", "true", "True"}
    timeout_seconds = int(args.timeout_seconds)
    env_cfg = read_env_config(SCRIPT_DIR.parent)
    fluent_root = env_cfg.get("fluent", r"C:\Program Files\ANSYS Inc\v231\fluent")
    mode = str(args.mode).strip().lower()
    if mode == "pyfluent":
        payload = _launch_pyfluent(fluent_root, timeout_seconds, show_gui)
    else:
        payload = _launch_cli(workspace, fluent_root, timeout_seconds, show_gui)
    payload["mode"] = mode
    payload["fluent_root"] = fluent_root
    write_json(workspace / "fluent_smoke_result.json", payload)
    print(str(workspace / "fluent_smoke_result.json"))


if __name__ == "__main__":
    main()
