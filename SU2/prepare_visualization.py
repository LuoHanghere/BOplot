from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import resolve_workspace, write_json


def build_visualization_manifest(workspace: Path) -> Path:
    static_candidates = ["mesh.vtk", "mesh.su2", "restart_flow.dat"]
    static_files = [workspace / name for name in static_candidates if (workspace / name).exists()]
    global_flow = sorted(workspace.glob("flow*.vtu")) + sorted(workspace.glob("flow*.pvtu"))
    surface_flow = sorted(workspace.glob("surface_flow*.vtu")) + sorted(workspace.glob("surface_flow*.pvtu"))
    existing_paths = static_files + global_flow + surface_flow
    existing = [str(path) for path in existing_paths]
    payload = {
        "workspace": str(workspace),
        "paraview_files": existing,
        "global_flow_files": [str(path) for path in global_flow],
        "surface_flow_files": [str(path) for path in surface_flow],
        "open_with_paraview": ["paraview {}".format(path) for path in existing],
    }
    output_path = workspace / "paraview_manifest.json"
    write_json(output_path, payload)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    workspace = resolve_workspace(args.workspace, base_dir=SCRIPT_DIR / "data")
    output_path = build_visualization_manifest(workspace)
    print(str(output_path))


if __name__ == "__main__":
    main()
