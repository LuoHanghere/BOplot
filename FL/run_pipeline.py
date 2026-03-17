from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_json, resolve_workspace, write_json
from mesh import build_mesh
from parametric_model import build_geometry
from run_fluent import run_solver


def run_all(
    input_path: Path,
    workspace: Path,
    mode: str,
    mesh_size: float,
    msh_version: float,
    mesh_binary: bool,
) -> Path:
    geometry_path = build_geometry(input_path, workspace)
    mesh_path = build_mesh(
        geometry_path,
        workspace,
        mesh_size,
        msh_version=msh_version,
        mesh_binary=mesh_binary,
    )
    config = read_json(input_path).get("fluent", {})
    if not isinstance(config, dict):
        config = {}
    output_path = run_solver(mesh_path, workspace, mode, config)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--mode", default="fluent")
    parser.add_argument("--mesh-size", default="0.02")
    parser.add_argument("--msh-version", default="2.2")
    parser.add_argument("--mesh-binary", default="0")
    args = parser.parse_args()
    data_dir = SCRIPT_DIR / "data"
    workspace = resolve_workspace(args.workspace, base_dir=data_dir)
    output_path = run_all(
        Path(args.input),
        workspace,
        str(args.mode).lower(),
        float(args.mesh_size),
        float(args.msh_version),
        str(args.mesh_binary).strip() in {"1", "true", "True"},
    )
    write_json(workspace / "pipeline_result.json", {"result": str(output_path)})
    print(str(output_path))


if __name__ == "__main__":
    main()
