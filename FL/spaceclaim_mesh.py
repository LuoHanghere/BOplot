from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_env_config, resolve_workspace, write_json


def _build_script(input_path: Path, scdoc_path: Path, export_path: Path | None) -> str:
    lines = [
        "import sys",
        "from SpaceClaim.Api.V23 import *",
        "from SpaceClaim.Api.V23.Extensibility import *",
        "part = DocumentOpen.Execute(r'{}')".format(input_path),
        "DocumentSave.Execute(r'{}')".format(scdoc_path),
    ]
    if export_path is not None:
        lines.append("DocumentSave.Execute(r'{}')".format(export_path))
    return "\n".join(lines)


def _resolve_export_path(workspace: Path, export_format: str) -> Path:
    fmt = export_format.strip().lower()
    if fmt in {"step", "stp"}:
        return workspace / "airfoil_from_spaceclaim.step"
    if fmt in {"iges", "igs"}:
        return workspace / "airfoil_from_spaceclaim.iges"
    if fmt == "none":
        return workspace / "airfoil_from_spaceclaim.none"
    raise ValueError("export_format must be step, iges, or none")


def run_spaceclaim(
    input_path: Path,
    workspace: Path,
    scdm_path: str,
    export_format: str,
    timeout_seconds: int,
) -> Path:
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    script_path = logs_dir / "spaceclaim_import.py"
    scdoc_path = workspace / "airfoil.scdoc"
    export_path = _resolve_export_path(workspace, export_format)
    if export_format.strip().lower() == "none":
        target_export: Path | None = None
    else:
        target_export = export_path
    script_path.write_text(
        _build_script(input_path, scdoc_path, target_export),
        encoding="utf-8",
    )
    cmd = [
        scdm_path,
        "/RunScript={}".format(str(script_path)),
        "/ExitAfterScript=true",
        "/Headless=true",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
        timeout = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        result = subprocess.CompletedProcess(cmd, returncode=124, stdout=stdout, stderr=stderr)
        timeout = True
    result_path = logs_dir / "spaceclaim_result.json"
    write_json(
        result_path,
        {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timeout": timeout,
            "script": str(script_path),
            "scdoc": str(scdoc_path),
            "exported_cad": "" if target_export is None else str(target_export),
            "input": str(input_path),
        },
    )
    expected = scdoc_path if target_export is None else target_export
    if result.returncode != 0 and not expected.exists():
        raise RuntimeError("SpaceClaim failed: {}".format(result.stderr.strip()))
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--export-format", default="step")
    parser.add_argument("--timeout-seconds", default="300")
    args = parser.parse_args()
    data_dir = SCRIPT_DIR / "data"
    workspace = resolve_workspace(args.workspace, base_dir=data_dir)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = workspace / args.input
    env_cfg = read_env_config(SCRIPT_DIR.parent)
    scdm_path = env_cfg.get("scdm", "")
    if not scdm_path:
        scdm_path = r"C:\Program Files\ANSYS Inc\v231\scdm\SpaceClaim.exe"
    if not Path(scdm_path).exists():
        raise RuntimeError("scdm path not found: {}".format(scdm_path))
    output_path = run_spaceclaim(
        input_path=input_path,
        workspace=workspace,
        scdm_path=scdm_path,
        export_format=str(args.export_format),
        timeout_seconds=int(args.timeout_seconds),
    )
    print(str(output_path))


if __name__ == "__main__":
    main()
