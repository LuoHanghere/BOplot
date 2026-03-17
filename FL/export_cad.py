from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_json, resolve_workspace


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


def export_cad(geometry_path: Path, workspace: Path, cad_format: str) -> Path:
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
    gmsh.model.add("airfoil_cad")

    def add_point(point: dict) -> int:
        return gmsh.model.occ.addPoint(float(point["x"]), float(point["y"]), 0.0)

    upper_points = [add_point(p) for p in upper]
    lower_points = [add_point(p) for p in lower]
    leading_point = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
    trailing_point = gmsh.model.occ.addPoint(1.0, 0.0, 0.0)

    upper_curve = gmsh.model.occ.addSpline([leading_point, *upper_points, trailing_point])
    lower_curve = gmsh.model.occ.addSpline([trailing_point, *reversed(lower_points), leading_point])
    airfoil_wire = gmsh.model.occ.addWire([upper_curve, lower_curve])

    p1 = gmsh.model.occ.addPoint(xmin, ymin, 0.0)
    p2 = gmsh.model.occ.addPoint(xmax, ymin, 0.0)
    p3 = gmsh.model.occ.addPoint(xmax, ymax, 0.0)
    p4 = gmsh.model.occ.addPoint(xmin, ymax, 0.0)
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)
    outer_wire = gmsh.model.occ.addWire([l1, l2, l3, l4])
    gmsh.model.occ.addPlaneSurface([outer_wire, airfoil_wire])
    gmsh.model.occ.synchronize()

    ext = cad_format.lower()
    if ext.startswith("."):
        ext = ext[1:]
    if ext not in {"step", "stp", "iges", "igs", "brep"}:
        raise ValueError("cad_format must be step, iges, or brep")
    output_name = "airfoil.{}".format(ext)
    output_path = workspace / output_name
    gmsh.write(str(output_path))
    gmsh.finalize()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--geometry", default="geometry.json")
    parser.add_argument("--format", default="step")
    args = parser.parse_args()
    data_dir = SCRIPT_DIR / "data"
    workspace = resolve_workspace(args.workspace, base_dir=data_dir)
    geometry_path = Path(args.geometry)
    if not geometry_path.is_absolute():
        geometry_path = workspace / args.geometry
    output_path = export_cad(geometry_path, workspace, str(args.format))
    print(str(output_path))


if __name__ == "__main__":
    main()
