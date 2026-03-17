from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_json, resolve_workspace, write_json


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _farfield_bounds(raw: dict) -> tuple[float, float, float, float]:
    xmin = _float_value(raw.get("xmin"), -5.0)
    xmax = _float_value(raw.get("xmax"), 5.0)
    ymin = _float_value(raw.get("ymin"), -5.0)
    ymax = _float_value(raw.get("ymax"), 5.0)
    return xmin, xmax, ymin, ymax


def build_mesh(
    geometry_path: Path,
    workspace: Path,
    mesh_size: float,
    msh_version: float = 2.2,
    mesh_binary: bool = False,
) -> Path:
    data = read_json(geometry_path)
    upper = data.get("upper", [])
    lower = data.get("lower", [])
    farfield = data.get("farfield", {})
    if not isinstance(upper, list) or not isinstance(lower, list) or len(upper) < 2 or len(lower) < 2:
        raise ValueError("geometry.json requires upper and lower point lists")
    xmin, xmax, ymin, ymax = _farfield_bounds(farfield if isinstance(farfield, dict) else {})

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("airfoil")

    def add_point(point: dict) -> int:
        return gmsh.model.geo.addPoint(
            float(point["x"]), float(point["y"]), 0.0, mesh_size
        )

    upper_points = [add_point(p) for p in upper]
    lower_points = [add_point(p) for p in lower]
    leading_point = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, mesh_size)
    trailing_point = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, mesh_size)

    upper_curve = gmsh.model.geo.addBSpline([leading_point, *upper_points, trailing_point])
    lower_curve = gmsh.model.geo.addBSpline([trailing_point, *reversed(lower_points), leading_point])
    airfoil_loop = gmsh.model.geo.addCurveLoop([upper_curve, lower_curve])

    p1 = gmsh.model.geo.addPoint(xmin, ymin, 0.0, mesh_size)
    p2 = gmsh.model.geo.addPoint(xmax, ymin, 0.0, mesh_size)
    p3 = gmsh.model.geo.addPoint(xmax, ymax, 0.0, mesh_size)
    p4 = gmsh.model.geo.addPoint(xmin, ymax, 0.0, mesh_size)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    farfield_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

    surface = gmsh.model.geo.addPlaneSurface([farfield_loop, airfoil_loop])
    gmsh.model.geo.synchronize()

    airfoil_group = gmsh.model.addPhysicalGroup(1, [upper_curve, lower_curve])
    gmsh.model.setPhysicalName(1, airfoil_group, "airfoil")
    inlet_group = gmsh.model.addPhysicalGroup(1, [l4])
    gmsh.model.setPhysicalName(1, inlet_group, "inlet")
    outlet_group = gmsh.model.addPhysicalGroup(1, [l2])
    gmsh.model.setPhysicalName(1, outlet_group, "outlet")
    farfield_group = gmsh.model.addPhysicalGroup(1, [l1, l3])
    gmsh.model.setPhysicalName(1, farfield_group, "farfield")
    surface_group = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, surface_group, "fluid_domain")

    gmsh.option.setNumber("Mesh.MshFileVersion", float(msh_version))
    gmsh.option.setNumber("Mesh.Binary", 1 if mesh_binary else 0)
    gmsh.model.mesh.generate(2)
    output_path = workspace / "mesh.msh"
    gmsh.write(str(output_path))
    gmsh.finalize()

    write_json(
        workspace / "mesh_info.json",
        {"mesh": str(output_path), "surface": int(surface_group)},
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--geometry", default="geometry.json")
    parser.add_argument("--mesh-size", default="0.02")
    parser.add_argument("--msh-version", default="2.2")
    parser.add_argument("--mesh-binary", default="0")
    args = parser.parse_args()
    data_dir = SCRIPT_DIR / "data"
    workspace = resolve_workspace(args.workspace, base_dir=data_dir)
    geometry_path = Path(args.geometry)
    if not geometry_path.is_absolute():
        geometry_path = workspace / args.geometry
    mesh_size = float(args.mesh_size)
    msh_version = float(args.msh_version)
    mesh_binary = str(args.mesh_binary).strip() in {"1", "true", "True"}
    mesh_path = build_mesh(
        geometry_path,
        workspace,
        mesh_size,
        msh_version=msh_version,
        mesh_binary=mesh_binary,
    )
    print(str(mesh_path))


if __name__ == "__main__":
    main()
