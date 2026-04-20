from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from ..initialization.config import RuntimeConfig, project_runtime_config
from ..objective.evaluator import Evaluator, MockQuadraticEvaluator, SubprocessEvaluator
from ..initialization.file_io import list_new_json_files, read_json, write_json
from .heartbeat import write_heartbeat
from ..initialization.models import TaskConfig, TrialInput
from ..initialization.task_config_store import load_task_config
from ..initialization.validator import ValidationError, validate_trial_input


def _output_filename(task_id: str, iteration: int) -> str:
    return "{}_iter_{:04d}_output.json".format(task_id, iteration)


def _error_filename(src_name: str) -> str:
    return "{}_error.json".format(Path(src_name).stem)


def _create_evaluator(task_cfg: TaskConfig) -> Evaluator:
    if task_cfg.problem == "branin":
        return SubprocessEvaluator(module="plat_bo.objective.branin_program")
    if task_cfg.problem == "su2_airfoil_2d":
        extra = {"problem_config": task_cfg.problem_config or {}}
        return SubprocessEvaluator(module="plat_bo.objective.su2_airfoil_program", extra_input=extra)
    
    # Generic subprocess evaluator using problem_config.module
    if task_cfg.problem_config and "module" in task_cfg.problem_config:
        return SubprocessEvaluator(module=task_cfg.problem_config["module"])
        
    return MockQuadraticEvaluator()


def _validate_against_task_config(parameters: list[float], bounds: list[list[float]]) -> None:
    if len(parameters) != len(bounds):
        raise ValidationError("parameters length mismatch with bounds dimension")
    for i, value in enumerate(parameters):
        lo, hi = bounds[i]
        if value < lo or value > hi:
            raise ValidationError(
                "parameter index {} out of bounds [{}, {}]".format(i, lo, hi)
            )


def run_worker(config: RuntimeConfig | None = None) -> None:
    cfg = config or RuntimeConfig()
    cfg.ensure_dirs()
    evaluator_cache: dict[str, Evaluator] = {}

    while True:
        write_heartbeat(cfg.heartbeat_dir, "worker")
        files = list_new_json_files(cfg.inbox_dir)
        if not files:
            time.sleep(cfg.poll_seconds)
            continue

        for src in files:
            try:
                try:
                    payload = read_json(src)
                except json.JSONDecodeError:
                    continue
                validate_trial_input(payload)
                trial = TrialInput.from_dict(payload)
                task_cfg = load_task_config(cfg.task_config_dir, trial.task_id)
                _validate_against_task_config(trial.parameters, task_cfg.bounds)
                if trial.task_id not in evaluator_cache:
                    evaluator_cache[trial.task_id] = _create_evaluator(task_cfg)
                evaluator = evaluator_cache[trial.task_id]
                result = evaluator.evaluate(trial)
                output_path = cfg.outbox_dir / _output_filename(trial.task_id, trial.iteration)
                write_json(output_path, result.to_dict())
            except (ValidationError, ValueError, KeyError, TypeError, FileNotFoundError) as exc:
                error_path = cfg.outbox_dir / _error_filename(src.name)
                write_json(
                    error_path,
                    {
                        "success": False,
                        "message": "input validation failed: {}".format(exc),
                        "source_file": src.name,
                    },
                )
            finally:
                target = cfg.processed_dir / src.name
                if target.exists():
                    target = cfg.processed_dir / "{}_{}{}".format(
                        src.stem, int(time.time()), src.suffix
                    )
                shutil.move(str(src), str(target))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run worker loop for a project.")
    parser.add_argument("--project-id", default="default")
    args = parser.parse_args()
    run_worker(config=project_runtime_config(args.project_id))
