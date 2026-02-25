from __future__ import annotations

from typing import Any


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
    required = {"task_id", "problem", "strategy", "bounds", "max_iterations"}
    missing = required.difference(data.keys())
    if missing:
        raise ValidationError(f"missing task config fields: {sorted(missing)}")

    if not isinstance(data["task_id"], str) or not data["task_id"].strip():
        raise ValidationError("task_id must be a non-empty string")

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
