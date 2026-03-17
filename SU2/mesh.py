from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_json, resolve_workspace, write_json

MESH_LEVEL_SIZES = {
    "low": 0.12,
    "medium": 0.06,
    "high": 0.03,
}


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _farfield_bounds(raw: dict[str, Any]) -> tuple[float, float, float, float]:
    xmin = _float_value(raw.get("xmin"), -8.0)
    xmax = _float_value(raw.get("xmax"), 12.0)
    ymin = _float_value(raw.get("ymin"), -8.0)
    ymax = _float_value(raw.get("ymax"), 8.0)
    return xmin, xmax, ymin, ymax


def _resolve_mesh_size(mesh_level: str, mesh_size: float | None) -> float:
    if mesh_size is not None and mesh_size > 0.0:
        return float(mesh_size)
    return float(MESH_LEVEL_SIZES.get(mesh_level, MESH_LEVEL_SIZES["medium"]))


def _leading_edge_points(
    upper: list[dict[str, Any]],
    lower: list[dict[str, Any]],
    mesh_size: float,
) -> list[int]:
    import gmsh

    x_u = _float_value(upper[0].get("x"), 0.1)
    y_u = _float_value(upper[0].get("y"), 0.02)
    x_l = _float_value(lower[0].get("x"), 0.1)
    y_l = _float_value(lower[0].get("y"), -0.01)
    x0 = max(1e-4, min(x_u, x_l))
    thickness = max(1e-4, y_u - y_l)
    bulge = min(0.06, max(0.02, 0.45 * thickness))
    ts = [0.2, 0.4, 0.6, 0.8]
    point_ids: list[int] = []
    for t in ts:
        y = y_l + (y_u - y_l) * t
        profile = 1.0 - (2.0 * t - 1.0) ** 2
        x = max(1e-5, x0 - bulge * profile)
        point_ids.append(gmsh.model.geo.addPoint(x, y, 0.0, mesh_size))
    return point_ids


def _leading_edge_center_and_radius(upper: list[dict[str, Any]], lower: list[dict[str, Any]]) -> tuple[float, float, float]:
    x_u = _float_value(upper[0].get("x"), 0.1)
    y_u = _float_value(upper[0].get("y"), 0.02)
    x_l = _float_value(lower[0].get("x"), 0.1)
    y_l = _float_value(lower[0].get("y"), -0.01)
    x0 = max(1e-5, min(x_u, x_l))
    y0 = 0.5 * (y_u + y_l)
    thickness = max(1e-4, y_u - y_l)
    radius = max(0.012, min(0.08, 0.7 * thickness + 0.35 * x0))
    return x0, y0, radius


def build_mesh(
    geometry_path: Path,
    workspace: Path,
    mesh_level: str = "medium",
    mesh_size: float | None = None,
    msh_version: float = 2.2,
    mesh_binary: bool = False,
) -> Path:
    data = read_json(geometry_path)
    upper = data.get("upper", [])
    lower = data.get("lower", [])
    farfield = data.get("farfield", {})
    if not isinstance(upper, list) or not isinstance(lower, list) or len(upper) < 2 or len(lower) < 2:
        raise ValueError("geometry.json requires upper and lower point lists")
    mesh_size_final = _resolve_mesh_size(mesh_level, mesh_size)
    xmin, xmax, ymin, ymax = _farfield_bounds(farfield if isinstance(farfield, dict) else {})

    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("airfoil_su2")

    def add_point(point: dict[str, Any]) -> int:
        return gmsh.model.geo.addPoint(float(point["x"]), float(point["y"]), 0.0, mesh_size_final)

    upper_points = [add_point(p) for p in upper]
    lower_points = [add_point(p) for p in lower]
    leading_curve_points = _leading_edge_points(upper, lower, mesh_size_final)
    trailing_point = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, mesh_size_final)

    upper_anchor = upper_points[0]
    lower_anchor = lower_points[0]
    upper_body = upper_points[1:]
    lower_body = lower_points[1:]
    upper_curve = gmsh.model.geo.addBSpline([upper_anchor, *upper_body, trailing_point])
    lower_curve = gmsh.model.geo.addBSpline([trailing_point, *reversed(lower_body), lower_anchor])
    leading_curve = gmsh.model.geo.addBSpline([lower_anchor, *leading_curve_points, upper_anchor])
    airfoil_loop = gmsh.model.geo.addCurveLoop([upper_curve, lower_curve, leading_curve])

    p1 = gmsh.model.geo.addPoint(xmin, ymin, 0.0, mesh_size_final)
    p2 = gmsh.model.geo.addPoint(xmax, ymin, 0.0, mesh_size_final)
    p3 = gmsh.model.geo.addPoint(xmax, ymax, 0.0, mesh_size_final)
    p4 = gmsh.model.geo.addPoint(xmin, ymax, 0.0, mesh_size_final)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    farfield_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

    surface = gmsh.model.geo.addPlaneSurface([farfield_loop, airfoil_loop])
    gmsh.model.geo.synchronize()

    airfoil_group = gmsh.model.addPhysicalGroup(1, [upper_curve, lower_curve, leading_curve])
    gmsh.model.setPhysicalName(1, airfoil_group, "airfoil")
    inlet_group = gmsh.model.addPhysicalGroup(1, [l4])
    gmsh.model.setPhysicalName(1, inlet_group, "inlet")
    outlet_group = gmsh.model.addPhysicalGroup(1, [l2])
    gmsh.model.setPhysicalName(1, outlet_group, "outlet")
    farfield_group = gmsh.model.addPhysicalGroup(1, [l1, l3])
    gmsh.model.setPhysicalName(1, farfield_group, "farfield")
    surface_group = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, surface_group, "fluid_domain")

    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", [upper_curve, lower_curve, leading_curve])
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", max(mesh_size_final * 0.2, 1e-4))
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", mesh_size_final * 1.5)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.02)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 0.9)
    x_le, y_le, le_radius = _leading_edge_center_and_radius(upper, lower)
    leading_ball = gmsh.model.mesh.field.add("Ball")
    gmsh.model.mesh.field.setNumber(leading_ball, "Radius", le_radius)
    gmsh.model.mesh.field.setNumber(leading_ball, "Thickness", 1.4 * le_radius)
    gmsh.model.mesh.field.setNumber(leading_ball, "XCenter", x_le)
    gmsh.model.mesh.field.setNumber(leading_ball, "YCenter", y_le)
    gmsh.model.mesh.field.setNumber(leading_ball, "ZCenter", 0.0)
    gmsh.model.mesh.field.setNumber(leading_ball, "VIn", max(mesh_size_final * 0.1, 8e-5))
    gmsh.model.mesh.field.setNumber(leading_ball, "VOut", mesh_size_final * 1.3)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field, leading_ball])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MshFileVersion", float(msh_version))
    gmsh.option.setNumber("Mesh.Binary", 1 if mesh_binary else 0)
    gmsh.model.mesh.generate(2)

    msh_path = workspace / "mesh.msh"
    su2_path = workspace / "mesh.su2"
    vtk_path = workspace / "mesh.vtk"
    gmsh.write(str(msh_path))
    gmsh.write(str(su2_path))
    gmsh.write(str(vtk_path))
    gmsh.finalize()

    write_json(
        workspace / "mesh_info.json",
        {
            "mesh_level": mesh_level,
            "mesh_size": mesh_size_final,
            "leading_edge_refine": {"x": x_le, "y": y_le, "radius": le_radius},
            "msh_file": str(msh_path),
            "su2_file": str(su2_path),
            "vtk_file": str(vtk_path),
            "surface_group": int(surface_group),
        },
    )
    return su2_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--geometry", default="geometry.json")
    parser.add_argument("--mesh-level", default="medium")
    parser.add_argument("--mesh-size", default="")
    parser.add_argument("--msh-version", default="2.2")
    parser.add_argument("--mesh-binary", default="0")
    args = parser.parse_args()
    data_dir = SCRIPT_DIR / "data"
    workspace = resolve_workspace(args.workspace, base_dir=data_dir)
    geometry_path = Path(args.geometry)
    if not geometry_path.is_absolute():
        geometry_path = workspace / args.geometry
    raw_mesh_size = str(args.mesh_size).strip()
    mesh_size = float(raw_mesh_size) if raw_mesh_size else None
    su2_path = build_mesh(
        geometry_path,
        workspace,
        mesh_level=str(args.mesh_level).strip().lower(),
        mesh_size=mesh_size,
        msh_version=float(args.msh_version),
        mesh_binary=str(args.mesh_binary).strip() in {"1", "true", "True"},
    )
    print(str(su2_path))


if __name__ == "__main__":
    main()
