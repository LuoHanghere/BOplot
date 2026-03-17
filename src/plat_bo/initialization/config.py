from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECTS_ROOT = SRC_ROOT.parent / "project"


@dataclass(frozen=True)
class RuntimeConfig:
    project_id: str = "default"
    inbox_dir: Path = Path("runtime/inbox")
    outbox_dir: Path = Path("runtime/outbox")
    processed_dir: Path = Path("runtime/processed")
    state_dir: Path = Path("runtime/state")
    task_config_dir: Path = Path("runtime/state/tasks")
    heartbeat_dir: Path = Path("runtime/state/heartbeat")
    logs_dir: Path = Path("runtime/state/logs")
    iterations_dir: Path = Path("runtime/state/iterations")
    poll_seconds: float = 1.0
    default_max_iterations: int = 20

    def ensure_dirs(self) -> None:
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.task_config_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.iterations_dir.mkdir(parents=True, exist_ok=True)


def project_runtime_config(
    project_id: str, projects_root: Path = DEFAULT_PROJECTS_ROOT
) -> RuntimeConfig:
    base = projects_root / project_id
    state = base / "state"
    return RuntimeConfig(
        project_id=project_id,
        inbox_dir=base / "inbox",
        outbox_dir=base / "outbox",
        processed_dir=base / "processed",
        state_dir=state,
        task_config_dir=state / "tasks",
        heartbeat_dir=state / "heartbeat",
        logs_dir=state / "logs",
        iterations_dir=state / "iterations",
    )
