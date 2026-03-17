from __future__ import annotations

import argparse

from .config import RuntimeConfig, project_runtime_config
from ..acquisition.strategy_config import resolve_component_config
from ..surrogate.engine import BOEngine
from .file_io import write_json
from .models import TaskConfig
from .task_config_store import save_task_config
from .validator import validate_task_config


def default_task_config(task_id: str = "branin-001") -> TaskConfig:
    return TaskConfig(
        task_id=task_id,
        problem="branin",
        strategy="base_single_task_gp_ei",
        bounds=[[-5.0, 10.0], [0.0, 15.0]],
        max_iterations=20,
        initial_random_trials=5,
        objective_direction="minimize",
        objective_target=None,
    )


def bootstrap_task(config: TaskConfig, runtime_cfg: RuntimeConfig | None = None) -> None:
    cfg = runtime_cfg or RuntimeConfig()
    cfg.ensure_dirs()
    strategy, component_config = resolve_component_config(config.strategy, config.component_config)
    config.strategy = strategy
    config.component_config = component_config
    validate_task_config(config.to_dict())
    save_task_config(cfg.task_config_dir, config)
    engine = BOEngine(state_dir=cfg.state_dir, task_config=config)
    first = engine.suggest_next(task_id=config.task_id, iteration=0)
    write_json(cfg.inbox_dir / "{}_iter_0000_input.json".format(config.task_id), first.to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap BO task.")
    parser.add_argument("--task-id", default="branin-001")
    parser.add_argument("--project-id", default="default")
    parser.add_argument("--max-iterations", type=int, default=20)
    args = parser.parse_args()
    cfg = default_task_config(task_id=args.task_id)
    cfg.max_iterations = args.max_iterations
    runtime_cfg = project_runtime_config(args.project_id)
    bootstrap_task(cfg, runtime_cfg=runtime_cfg)


if __name__ == "__main__":
    main()
