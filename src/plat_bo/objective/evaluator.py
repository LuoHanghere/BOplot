from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

from ..initialization.models import TrialInput, TrialOutput


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, trial: TrialInput) -> TrialOutput:
        raise NotImplementedError


class MockQuadraticEvaluator(Evaluator):
    """Deterministic baseline for framework smoke tests."""

    def evaluate(self, trial: TrialInput) -> TrialOutput:
        start = time.perf_counter()
        obj = sum((x - 0.5) ** 2 for x in trial.parameters)
        if any(x < 0.0 or x > 1.0 for x in trial.parameters):
            return TrialOutput(
                task_id=trial.task_id,
                iteration=trial.iteration,
                parameters=trial.parameters,
                objective=float("nan"),
                success=False,
                message="parameter out of range [0,1]",
                cost_seconds=time.perf_counter() - start,
            )

        return TrialOutput(
            task_id=trial.task_id,
            iteration=trial.iteration,
            parameters=trial.parameters,
            objective=obj,
            success=True,
            message="ok",
            cost_seconds=time.perf_counter() - start,
        )


class SubprocessEvaluator(Evaluator):
    """
    Call an external program to evaluate objective.
    External program protocol:
    - input file JSON: {"task_id","iteration","parameters"}
    - output file JSON: {"objective": float, "success": bool, "message": str}
    """

    def __init__(self, module: str, extra_input: dict | None = None) -> None:
        self.module = module
        self.extra_input = dict(extra_input or {})

    def evaluate(self, trial: TrialInput) -> TrialOutput:
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="plat_bo_eval_") as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.json"
            output_path = tmp / "output.json"
            payload = trial.to_dict()
            payload.update(self.extra_input)
            with input_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            cmd = [
                sys.executable,
                "-m",
                self.module,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                with output_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                return TrialOutput(
                    task_id=trial.task_id,
                    iteration=trial.iteration,
                    parameters=trial.parameters,
                    objective=float(payload["objective"]),
                    success=bool(payload.get("success", True)),
                    message=str(payload.get("message", "ok")),
                    cost_seconds=time.perf_counter() - start,
                    objective_vector=(
                        [float(v) for v in payload["objective_vector"]]
                        if isinstance(payload.get("objective_vector"), list)
                        else None
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                return TrialOutput(
                    task_id=trial.task_id,
                    iteration=trial.iteration,
                    parameters=trial.parameters,
                    objective=float("nan"),
                    success=False,
                    message="subprocess evaluator failed: {}".format(exc),
                    cost_seconds=time.perf_counter() - start,
                )
