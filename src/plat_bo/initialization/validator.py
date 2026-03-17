from __future__ import annotations

from typing import Any

from ..acquisition.strategy_config import resolve_component_config


class ValidationError(ValueError):
    pass


def validate_trial_input(data: dict[str, Any]) -> None:
    required = {"task_id", "iteration", "parameters"}
    missing = required.difference(data.keys())
    if missing:
        raise ValidationError(f"missing fields: {sorted(missing)}")

    if not isinstance(data["task_id"], str) or not data["task_id"].strip():
        raise ValidationError("task_id must be a non-empty string")

    if not isinstance(data["iteration"], int) or data["iteration"] < 0:
        raise ValidationError("iteration must be a non-negative integer")

    if not isinstance(data["parameters"], list) or not data["parameters"]:
        raise ValidationError("parameters must be a non-empty list")

    for value in data["parameters"]:
        if not isinstance(value, (int, float)):
            raise ValidationError("parameters must be numeric")


def validate_task_config(data: dict[str, Any]) -> None:
    required = {"task_id", "problem", "bounds", "max_iterations"}
    missing = required.difference(data.keys())
    if missing:
        raise ValidationError(f"missing task config fields: {sorted(missing)}")

    if not isinstance(data["task_id"], str) or not data["task_id"].strip():
        raise ValidationError("task_id must be a non-empty string")

    strategy = data.get("strategy")
    if strategy is not None and (not isinstance(strategy, str) or not strategy.strip()):
        raise ValidationError("strategy must be a non-empty string when provided")
    if strategy is None and data.get("component_config") is None:
        raise ValidationError("either strategy or component_config must be provided")

    raw_component_config = data.get("component_config")
    if raw_component_config is not None and not isinstance(raw_component_config, dict):
        raise ValidationError("component_config must be an object when provided")
    raw_problem_config = data.get("problem_config")
    if raw_problem_config is not None and not isinstance(raw_problem_config, dict):
        raise ValidationError("problem_config must be an object when provided")
    try:
        resolve_component_config(
            None if strategy is None else str(strategy),
            None if raw_component_config is None else raw_component_config,
        )
    except ValueError as exc:
        raise ValidationError(str(exc))

    if not isinstance(data["bounds"], list) or not data["bounds"]:
        raise ValidationError("bounds must be a non-empty list")

    for pair in data["bounds"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValidationError("each bounds item must be [low, high]")
        lo, hi = pair
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or lo >= hi:
            raise ValidationError("bounds must be numeric and low < high")

    if "objective_target" in data and data["objective_target"] is not None:
        if not isinstance(data["objective_target"], (int, float)):
            raise ValidationError("objective_target must be numeric or null")
