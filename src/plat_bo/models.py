from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrialInput:
    task_id: str
    iteration: int
    parameters: list[float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialInput":
        return cls(
            task_id=str(data["task_id"]),
            iteration=int(data["iteration"]),
            parameters=[float(x) for x in data["parameters"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrialOutput:
    task_id: str
    iteration: int
    parameters: list[float]
    objective: float
    success: bool
    message: str
    cost_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskConfig:
    task_id: str
    problem: str
    strategy: str
    bounds: list[list[float]]
    max_iterations: int
    initial_random_trials: int
    objective_direction: str = "minimize"
    objective_target: float | None = None

    @property
    def dim(self) -> int:
        return len(self.bounds)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskConfig":
        return cls(
            task_id=str(data["task_id"]),
            problem=str(data["problem"]),
            strategy=str(data["strategy"]),
            bounds=[[float(pair[0]), float(pair[1])] for pair in data["bounds"]],
            max_iterations=int(data["max_iterations"]),
            initial_random_trials=int(data.get("initial_random_trials", 5)),
            objective_direction=str(data.get("objective_direction", "minimize")),
            objective_target=(
                None if data.get("objective_target") is None else float(data.get("objective_target"))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
