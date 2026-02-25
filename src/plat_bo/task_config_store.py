from __future__ import annotations

from pathlib import Path

from .file_io import read_json, write_json
from .models import TaskConfig
from .validator import validate_task_config


def task_config_path(task_config_dir: Path, task_id: str) -> Path:
    return task_config_dir / "{}.json".format(task_id)


def save_task_config(task_config_dir: Path, config: TaskConfig) -> None:
    path = task_config_path(task_config_dir, config.task_id)
    write_json(path, config.to_dict())


def load_task_config(task_config_dir: Path, task_id: str) -> TaskConfig:
    path = task_config_path(task_config_dir, task_id)
    if not path.exists():
        raise FileNotFoundError("task config not found for task_id={}".format(task_id))
    data = read_json(path)
    validate_task_config(data)
    return TaskConfig.from_dict(data)

