from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_json, resolve_workspace, write_json


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_su2_config(input_data: dict[str, Any]) -> dict[str, Any]:
    su2_cfg = input_data.get("su2", {})
    if isinstance(su2_cfg, dict):
        return su2_cfg
    return {}


def build_case_config(
    workspace: Path,
    input_data: dict[str, Any],
    mesh_name: str = "mesh.su2",
    iterations_override: int | None = None,
) -> Path:
    cfg = _get_su2_config(input_data)
    solver = str(cfg.get("solver", "EULER")).strip()
    aoa_deg = _safe_float(cfg.get("aoa_deg"), 2.0)
    mach = _safe_float(cfg.get("mach"), 0.5)
    reynolds = _safe_float(cfg.get("reynolds"), 1_000_000.0)
    iterations = _safe_int(cfg.get("iterations"), 800)
    turbulence_model = str(cfg.get("turbulence_model", "SST")).strip()
    freestream_temp = _safe_float(cfg.get("freestream_temperature"), 288.15)
    freestream_press = _safe_float(cfg.get("freestream_pressure"), 101325.0)
    turbulent_intensity = _safe_float(cfg.get("turbulence_intensity"), 0.05)
    turb2lam_ratio = _safe_float(cfg.get("turbulence_viscosity_ratio"), 10.0)
    boundary_mode = str(cfg.get("boundary_mode", "inlet_outlet")).strip().lower()
    viscous = solver.upper() in {"RANS", "NAVIER_STOKES"}
    cfl_default = 0.5 if viscous else 1.0
    cfl_number = _safe_float(cfg.get("cfl_number"), cfl_default)
    cfl_adapt = str(cfg.get("cfl_adapt", "NO")).strip().upper()
    if iterations_override is not None and iterations_override > 0:
        iterations = int(iterations_override)
    gamma = 1.4
    gas_constant = 287.058
    speed_of_sound = math.sqrt(max(1e-8, gamma * gas_constant * freestream_temp))
    freestream_velocity = mach * speed_of_sound
    aoa_rad = math.radians(aoa_deg)
    velocity_x = freestream_velocity * math.cos(aoa_rad)
    velocity_y = freestream_velocity * math.sin(aoa_rad)

    lines = [
        "SOLVER= {}".format(solver),
        "MATH_PROBLEM= DIRECT",
        "RESTART_SOL= NO",
        "SYSTEM_MEASUREMENTS= SI",
        "MACH_NUMBER= {:.8f}".format(mach),
        "AOA= {:.8f}".format(aoa_deg),
        "SIDESLIP_ANGLE= 0.0",
        "REYNOLDS_NUMBER= {:.8f}".format(reynolds),
        "REYNOLDS_LENGTH= 1.0",
        "INIT_OPTION= REYNOLDS",
        "FREESTREAM_OPTION= TEMPERATURE_FS",
        "FREESTREAM_VELOCITY= ( {:.8f}, {:.8f}, 0.0 )".format(velocity_x, velocity_y),
        "FREESTREAM_TEMPERATURE= {:.8f}".format(freestream_temp),
        "FREESTREAM_PRESSURE= {:.8f}".format(freestream_press),
        "REF_ORIGIN_MOMENT_X= 0.25",
        "REF_ORIGIN_MOMENT_Y= 0.00",
        "REF_ORIGIN_MOMENT_Z= 0.00",
        "REF_LENGTH= 1.0",
        "REF_AREA= 1.0",
        "MARKER_HEATFLUX= ( airfoil, 0.0 )" if viscous else "MARKER_EULER= ( airfoil )",
        "MARKER_MONITORING= ( airfoil )",
        "MARKER_PLOTTING= ( airfoil )",
        "CONV_NUM_METHOD_FLOW= JST",
        "MUSCL_FLOW= NO",
        "NUM_METHOD_GRAD= GREEN_GAUSS",
        "CFL_NUMBER= {:.8f}".format(cfl_number),
        "CFL_ADAPT= {}".format(cfl_adapt),
        "ITER= {}".format(iterations),
        "LINEAR_SOLVER= FGMRES",
        "LINEAR_SOLVER_PREC= ILU",
        "LINEAR_SOLVER_ITER= 8",
        "LINEAR_SOLVER_ERROR= 1E-6",
        "CONV_FIELD= RMS_DENSITY",
        "CONV_RESIDUAL_MINVAL= -8",
        "CONV_STARTITER= 20",
        "MESH_FILENAME= {}".format(mesh_name),
        "MESH_FORMAT= SU2",
        "TABULAR_FORMAT= CSV",
        "CONV_FILENAME= history",
        "RESTART_FILENAME= restart_flow.dat",
        "VOLUME_FILENAME= flow",
        "SURFACE_FILENAME= surface_flow",
        "OUTPUT_FILES= ( RESTART, PARAVIEW, SURFACE_PARAVIEW, CSV )",
        "SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, LIFT, DRAG )",
        "HISTORY_OUTPUT= ( ITER, RMS_RES, LIFT, DRAG, MOMENT_X )",
    ]
    if boundary_mode == "inlet_outlet":
        mach_safe = max(0.0, mach)
        t0 = freestream_temp * (1.0 + 0.5 * (gamma - 1.0) * mach_safe * mach_safe)
        p0 = freestream_press * (1.0 + 0.5 * (gamma - 1.0) * mach_safe * mach_safe) ** (gamma / (gamma - 1.0))
        dir_x = math.cos(aoa_rad)
        dir_y = math.sin(aoa_rad)
        lines.extend(
            [
                "MARKER_INLET= ( inlet, {:.8f}, {:.8f}, {:.8f}, {:.8f}, 0.0 )".format(
                    t0, p0, dir_x, dir_y
                ),
                "MARKER_OUTLET= ( outlet, {:.8f} )".format(freestream_press),
                "MARKER_NEARFIELD= ( farfield )",
            ]
        )
    else:
        lines.append("MARKER_NEARFIELD= ( farfield, inlet, outlet )")
    if viscous:
        lines.extend(
            [
                "KIND_TURB_MODEL= {}".format(turbulence_model),
                "CONV_NUM_METHOD_TURB= SCALAR_UPWIND",
                "MUSCL_TURB= NO",
                "FREESTREAM_TURBULENCEINTENSITY= {:.8f}".format(turbulent_intensity),
                "FREESTREAM_TURB2LAMVISCRATIO= {:.8f}".format(turb2lam_ratio),
            ]
        )

    case_cfg_path = workspace / "case.cfg"
    case_cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        workspace / "case_setup.json",
        {
            "solver": solver,
            "aoa_deg": aoa_deg,
            "mach": mach,
            "reynolds": reynolds,
            "iterations": iterations,
            "mesh": mesh_name,
            "config_file": str(case_cfg_path),
        },
    )
    return case_cfg_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--mesh", default="mesh.su2")
    parser.add_argument("--iterations", default="")
    args = parser.parse_args()
    data_dir = SCRIPT_DIR / "data"
    workspace = resolve_workspace(args.workspace, base_dir=data_dir)
    input_data = read_json(Path(args.input))
    raw_iterations = str(args.iterations).strip()
    iterations = int(raw_iterations) if raw_iterations else None
    case_cfg_path = build_case_config(
        workspace=workspace,
        input_data=input_data,
        mesh_name=str(args.mesh),
        iterations_override=iterations,
    )
    print(str(case_cfg_path))


if __name__ == "__main__":
    main()
