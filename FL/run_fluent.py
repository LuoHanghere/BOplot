from __future__ import annotations

import argparse
import math
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
    cl = 0.8 + 2.0 * camber
    cd = 0.02 + 0.4 * camber * camber
    return {
        "lift_coefficient": cl,
        "drag_coefficient": cd,
        "objective": -(cl / max(cd, 1e-6)),
        "objective_vector": [-cl, cd],
        "success": True,
        "message": "mock",
    }


def _default_config() -> dict[str, Any]:
    return {
        "processors": 1,
        "iterations": 300,
        "freestream_velocity": 30.0,
        "aoa_deg": 2.0,
        "density": 1.225,
        "viscosity": 1.7894e-5,
        "turbulence_model": "k-omega-sst",
        "turbulence_intensity": 0.05,
        "turbulence_length_scale": 0.1,
        "inlet_name": "inlet",
        "outlet_name": "outlet",
        "farfield_name": "farfield",
        "airfoil_name": "airfoil",
        "use_udf": True,
        "udf_lib": "",
        "udf_functions": {},
    }


def _merge_config(raw: dict[str, Any]) -> dict[str, Any]:
    merged = _default_config()
    for key, value in raw.items():
        merged[key] = value
    return merged


def _try_set(obj: object, attr: str, value: object, errors: list[str]) -> None:
    try:
        if hasattr(obj, attr):
            setattr(obj, attr, value)
    except Exception as exc:
        errors.append("{}: {}".format(attr, exc))


def _get_group_item(group: object, name: str) -> object | None:
    try:
        return group[name]
    except Exception:
        return None


def _apply_settings(session: Any, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    models = session.setup.models
    viscous = models.viscous
    model = str(config.get("turbulence_model", "k-omega-sst")).lower()
    if "k-omega" in model:
        _try_set(viscous, "model", "k-omega", errors)
        if "sst" in model:
            _try_set(viscous, "k_omega_model", "sst", errors)
    elif "k-epsilon" in model:
        _try_set(viscous, "model", "k-epsilon", errors)
    else:
        _try_set(viscous, "model", "laminar", errors)

    materials = session.setup.materials
    air = _get_group_item(materials.fluid, "air")
    if air is not None:
        _try_set(air, "density", float(_safe_float(config.get("density"), 1.225)), errors)
        _try_set(air, "viscosity", float(_safe_float(config.get("viscosity"), 1.7894e-5)), errors)

    bc = session.setup.boundary_conditions
    inlet_name = str(config.get("inlet_name", "inlet"))
    outlet_name = str(config.get("outlet_name", "outlet"))
    farfield_name = str(config.get("farfield_name", "farfield"))
    airfoil_name = str(config.get("airfoil_name", "airfoil"))

    velocity = float(_safe_float(config.get("freestream_velocity"), 30.0))
    aoa = math.radians(float(_safe_float(config.get("aoa_deg"), 2.0)))
    vx = velocity * math.cos(aoa)
    vy = velocity * math.sin(aoa)

    inlet = _get_group_item(bc.velocity_inlet, inlet_name)
    if inlet is not None:
        _try_set(inlet, "vmag", velocity, errors)
        _try_set(inlet, "vx", vx, errors)
        _try_set(inlet, "vy", vy, errors)
        _try_set(inlet, "turbulence_intensity", float(_safe_float(config.get("turbulence_intensity"), 0.05)), errors)
        _try_set(inlet, "turbulence_length_scale", float(_safe_float(config.get("turbulence_length_scale"), 0.1)), errors)

    outlet = _get_group_item(bc.pressure_outlet, outlet_name)
    if outlet is not None:
        _try_set(outlet, "gauge_pressure", 0.0, errors)

    farfield = _get_group_item(bc.pressure_far_field, farfield_name)
    if farfield is not None:
        _try_set(farfield, "mach_number", 0.2, errors)
        _try_set(farfield, "temperature", 288.15, errors)

    airfoil = _get_group_item(bc.wall, airfoil_name)
    if airfoil is not None:
        _try_set(airfoil, "shear_condition", "no-slip", errors)

    if bool(config.get("use_udf")) and str(config.get("udf_lib", "")).strip():
        udf_lib = str(config.get("udf_lib"))
        try:
            session.setup.user_defined.library.load(udf_lib)
        except Exception as exc:
            errors.append("udf_load: {}".format(exc))
        udf_functions = config.get("udf_functions", {})
        if isinstance(udf_functions, dict):
            for key, value in udf_functions.items():
                try:
                    target = str(key)
                    session.tui.define.user_defined.function_hooks(target, str(value))
                except Exception as exc:
                    errors.append("udf_hook {}: {}".format(key, exc))

    return errors


def _run_fluent(mesh_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    import ansys.fluent.core as pyfluent

    processors = int(_safe_float(config.get("processors"), 1))
    iterations = int(_safe_float(config.get("iterations"), 300))
    launch_kwargs = {
        "precision": "double",
        "processor_count": processors,
        "mode": "solver",
        "show_gui": False,
    }
    fluent_path = config.get("fluent_path")
    if fluent_path:
        launch_kwargs["fluent_path"] = str(fluent_path)
    start = time.perf_counter()
    launch_errors: list[str] = []
    session = None
    launch_variants = [dict(launch_kwargs)]
    if "fluent_path" in launch_kwargs:
        fallback = dict(launch_kwargs)
        fallback.pop("fluent_path", None)
        launch_variants.append(fallback)
    for _ in range(3):
        for variant in launch_variants:
            try:
                session = pyfluent.launch_fluent(**variant)
                break
            except Exception as exc:
                launch_errors.append(str(exc))
        if session is not None:
            break
        time.sleep(2.0)
    if session is None:
        raise RuntimeError("launch failed: {}".format(" | ".join(launch_errors[-4:])))
    try:
        session.file.read_mesh(file_name=str(mesh_path))
        setting_errors = _apply_settings(session, config)
        session.solution.initialization.hybrid_initialize()
        session.solution.run_calculation.iterate(iter_count=iterations)
        reports = session.solution.report_definitions
        cl_value = reports.compute(report_defs=["lift-coeff"])
        cd_value = reports.compute(report_defs=["drag-coeff"])
        cl = float(cl_value[0].get("lift-coeff"))
        cd = float(cd_value[0].get("drag-coeff"))
        cost = time.perf_counter() - start
        return {
            "lift_coefficient": cl,
            "drag_coefficient": cd,
            "objective": -(cl / max(cd, 1e-6)),
            "objective_vector": [-cl, cd],
            "success": True,
            "message": "ok",
            "cost_seconds": cost,
            "setting_errors": setting_errors,
        }
    finally:
        session.exit()


def run_solver(mesh_path: Path, workspace: Path, mode: str, config: dict[str, Any]) -> Path:
    output_path = workspace / "result.json"
    if mode == "mock":
        payload = _mock_result(workspace)
    else:
        if not mesh_path.exists():
            payload = {
                "objective": float("nan"),
                "success": False,
                "message": "mesh not found: {}".format(mesh_path),
            }
        else:
            try:
                payload = _run_fluent(mesh_path, _merge_config(config))
            except Exception as exc:
                payload = {
                    "objective": float("nan"),
                    "success": False,
                    "message": "fluent failed: {}".format(exc),
                }
    write_json(output_path, payload)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--mesh", default="mesh.msh")
    parser.add_argument("--mode", default="fluent")
    parser.add_argument("--iterations", default="300")
    parser.add_argument("--processors", default="4")
    args = parser.parse_args()
    data_dir = SCRIPT_DIR / "data"
    workspace = resolve_workspace(args.workspace, base_dir=data_dir)
    mesh_path = Path(args.mesh)
    if not mesh_path.is_absolute():
        mesh_path = workspace / args.mesh
    config = {
        "iterations": int(args.iterations),
        "processors": int(args.processors),
    }
    env_cfg = read_env_config(SCRIPT_DIR.parent)
    fluent_path = env_cfg.get("fluent")
    if not fluent_path:
        fluent_path = r"C:\Program Files\ANSYS Inc\v231\fluent"
    config["fluent_path"] = fluent_path
    mode = str(args.mode).lower()
    output_path = run_solver(mesh_path, workspace, mode, config)
    print(str(output_path))


if __name__ == "__main__":
    main()
