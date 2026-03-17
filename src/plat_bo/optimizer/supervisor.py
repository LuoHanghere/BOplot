from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from ..initialization.config import RuntimeConfig, project_runtime_config
from ..surrogate.engine import BOEngine
from ..initialization.file_io import list_new_json_files, read_json, write_json
from .heartbeat import write_heartbeat
from ..initialization.models import TrialOutput
from ..initialization.task_config_store import load_task_config


def _is_iteration_output(path: Path) -> bool:
    return "_iter_" in path.stem and path.stem.endswith("_output")


def _next_input_filename(task_id: str, iteration: int) -> str:
    return "{}_iter_{:04d}_input.json".format(task_id, iteration)


def run_supervisor(config: RuntimeConfig | None = None) -> None:
    cfg = config or RuntimeConfig()
    cfg.ensure_dirs()
    handled: set[str] = set()
    engine_cache: dict[str, BOEngine] = {}

    while True:
        write_heartbeat(cfg.heartbeat_dir, "supervisor")
        files = list_new_json_files(cfg.outbox_dir)
        if not files:
            time.sleep(cfg.poll_seconds)
            continue

        for path in files:
            if path.name in handled:
                continue

            try:
                payload = read_json(path)
            except json.JSONDecodeError:
                continue
            if not _is_iteration_output(path):
                handled.add(path.name)
                continue

            try:
                objective_vector = payload.get("objective_vector")
                parsed_objective_vector = None
                if isinstance(objective_vector, list) and objective_vector:
                    parsed_objective_vector = [float(v) for v in objective_vector]
                output = TrialOutput(
                    task_id=str(payload["task_id"]),
                    iteration=int(payload["iteration"]),
                    parameters=[float(v) for v in payload["parameters"]],
                    objective=float(payload["objective"]),
                    success=bool(payload["success"]),
                    message=str(payload["message"]),
                    cost_seconds=float(payload["cost_seconds"]),
                    objective_vector=parsed_objective_vector,
                )
            except (KeyError, TypeError, ValueError):
                archived = cfg.processed_dir / path.name
                if archived.exists():
                    archived = cfg.processed_dir / "{}_{}{}".format(
                        path.stem, int(time.time()), path.suffix
                    )
                shutil.move(str(path), str(archived))
                handled.add(path.name)
                continue

            if output.task_id not in engine_cache:
                task_cfg = load_task_config(cfg.task_config_dir, output.task_id)
                engine_cache[output.task_id] = BOEngine(state_dir=cfg.state_dir, task_config=task_cfg)

            engine = engine_cache[output.task_id]
            engine.update(output)
            next_iter = output.iteration + 1
            stop_reason = None
            best = engine.best_objective(output.task_id)
            if next_iter > engine.task_config.max_iterations:
                stop_reason = "max_iterations_reached"
            elif (
                engine.task_config.objective_target is not None
                and best is not None
                and best <= float(engine.task_config.objective_target)
            ):
                stop_reason = "objective_target_reached"

            next_trial = None
            if stop_reason is None:
                next_trial = engine.suggest_next(task_id=output.task_id, iteration=next_iter)
                next_input_path = cfg.inbox_dir / _next_input_filename(output.task_id, next_iter)
                write_json(next_input_path, next_trial.to_dict())
            write_json(
                cfg.iterations_dir / "{}_iter_{:04d}_step.json".format(output.task_id, output.iteration),
                {
                    "task_id": output.task_id,
                    "iteration": output.iteration,
                    "input_parameters": output.parameters,
                    "objective": output.objective,
                    "objective_vector": output.objective_vector,
                    "success": output.success,
                    "message": output.message,
                    "next_iteration": None if next_trial is None else next_iter,
                    "next_parameters": None if next_trial is None else next_trial.parameters,
                    "best_objective": best,
                    "stop_reason": stop_reason,
                },
            )

            archived = cfg.processed_dir / path.name
            if archived.exists():
                archived = cfg.processed_dir / "{}_{}{}".format(
                    path.stem, int(time.time()), path.suffix
                )
            shutil.move(str(path), str(archived))
            handled.add(path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run supervisor loop for a project.")
    parser.add_argument("--project-id", default="default")
    args = parser.parse_args()
    run_supervisor(config=project_runtime_config(args.project_id))
