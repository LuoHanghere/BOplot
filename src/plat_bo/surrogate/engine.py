from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Tuple

import torch

from ..acquisition.strategy_config import resolve_component_config
from ..initialization.file_io import read_json, write_json
from ..initialization.models import TaskConfig, TrialInput, TrialOutput
from ..acquisition.botorch_acq import suggest_botorch
from ..acquisition.turbo_skeleton import suggest_turbo_skeleton


@dataclass
class EngineState:
    task_id: str
    history: List[dict[str, Any]]
    turbo_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BOEngine:
    def __init__(self, state_dir: Path, task_config: TaskConfig) -> None:
        self.state_file = state_dir / "{}_engine_state.json".format(task_config.task_id)
        self.task_config = task_config
        seed = sum(ord(ch) for ch in task_config.task_id) % (2**31 - 1)
        self._sobol = torch.quasirandom.SobolEngine(
            dimension=max(1, int(task_config.dim)),
            scramble=True,
            seed=seed,
        )

    def _load_or_init(self, task_id: str) -> EngineState:
        if self.state_file.exists():
            data = read_json(self.state_file)
            if data.get("task_id") == task_id:
                return EngineState(
                    task_id=task_id,
                    history=list(data.get("history", [])),
                    turbo_state=dict(data.get("turbo_state", {})),
                )
        return EngineState(task_id=task_id, history=[], turbo_state={})

    def _save(self, state: EngineState) -> None:
        write_json(self.state_file, state.to_dict())

    def get_state(self, task_id: str) -> EngineState:
        return self._load_or_init(task_id)

    def best_objective(self, task_id: str) -> float | None:
        state = self._load_or_init(task_id)
        vals: list[float] = []
        for item in state.history:
            if bool(item.get("success")):
                try:
                    vals.append(float(item["objective"]))
                except Exception:  # noqa: BLE001
                    pass
        if not vals:
            return None
        return min(vals)

    def update(self, output: TrialOutput) -> None:
        state = self._load_or_init(output.task_id)
        state.history.append(output.to_dict())
        self._update_turbo_state(state, output)
        self._save(state)

    def suggest_next(self, task_id: str, iteration: int) -> TrialInput:
        state = self._load_or_init(task_id)
        if len(state.history) < self.task_config.initial_random_trials:
            params = self._sample_sobol()
        else:
            params = self._suggest_botorch_or_fallback(state)
        return TrialInput(task_id=task_id, iteration=iteration, parameters=params)

    def _sample_sobol(self) -> list[float]:
        vec = self._sobol.draw(1)[0].tolist()
        params: list[float] = []
        for i, (lo, hi) in enumerate(self.task_config.bounds):
            u = float(vec[i]) if i < len(vec) else random.random()
            params.append(round(lo + (hi - lo) * u, 6))
        return params

    def _sample_uniform(self) -> list[float]:
        params: list[float] = []
        for lo, hi in self.task_config.bounds:
            params.append(round(lo + (hi - lo) * random.random(), 6))
        return params

    def _extract_train_data(self, state: EngineState) -> Tuple[list[list[float]], list[float], list[list[float]]]:
        xs: list[list[float]] = []
        ys: list[float] = []
        y_vecs: list[list[float]] = []
        for item in state.history:
            if bool(item.get("success")):
                try:
                    xs.append([float(v) for v in item["parameters"]])
                    ys.append(float(item["objective"]))
                    if "objective_vector" in item and item["objective_vector"] is not None:
                        y_vecs.append([float(v) for v in item["objective_vector"]])
                except Exception:  # noqa: BLE001
                    continue
        return xs, ys, y_vecs

    def _init_turbo_state(self, state: EngineState) -> None:
        if state.turbo_state:
            return
        state.turbo_state = {
            "length": 1.0,
            "success_counter": 0,
            "failure_counter": 0,
            "best_objective": None,
            "last_update_iteration": -1,
        }

    def _is_better(self, current: float, previous: float) -> bool:
        if self.task_config.objective_direction == "maximize":
            return current > previous
        return current < previous

    def _update_turbo_state(self, state: EngineState, output: TrialOutput) -> None:
        self._init_turbo_state(state)
        if not output.success:
            return
        try:
            objective = float(output.objective)
        except (TypeError, ValueError):
            return
        turbo = state.turbo_state
        best_raw = turbo.get("best_objective")
        improved = False
        if best_raw is None:
            improved = True
        else:
            improved = self._is_better(objective, float(best_raw))
        if improved:
            turbo["best_objective"] = objective
            turbo["success_counter"] = int(turbo.get("success_counter", 0)) + 1
            turbo["failure_counter"] = 0
        else:
            turbo["failure_counter"] = int(turbo.get("failure_counter", 0)) + 1
            turbo["success_counter"] = 0
        length = float(turbo.get("length", 1.0))
        success_counter = int(turbo.get("success_counter", 0))
        failure_counter = int(turbo.get("failure_counter", 0))
        if success_counter >= 3:
            length = min(1.6, length * 1.5)
            turbo["success_counter"] = 0
        if failure_counter >= 3:
            length = max(0.1, length / 2.0)
            turbo["failure_counter"] = 0
        turbo["length"] = round(length, 6)
        turbo["last_update_iteration"] = int(output.iteration)

    def _suggest_botorch_or_fallback(self, state: EngineState) -> list[float]:
        x_train, y_train, y_vec_train = self._extract_train_data(state)
        if len(x_train) < max(3, self.task_config.initial_random_trials):
            return self._sample_uniform()
        try:
            strategy, components = resolve_component_config(
                self.task_config.strategy, self.task_config.component_config
            )
            self.task_config.strategy = strategy
            self.task_config.component_config = components
            if components["surrogate_model"] == "turbo_trust_region_skeleton":
                self._init_turbo_state(state)
                trust_region_length = float(state.turbo_state.get("length", 1.0))
                return suggest_turbo_skeleton(
                    x_train, y_train, self.task_config, trust_region_length=trust_region_length
                )
            
            # Pass y_vec_train to support MOO
            return suggest_botorch(x_train, y_train, self.task_config, components, y_vec_train=y_vec_train)
        except Exception as e:
            print(f"Fallback due to exception: {e}")
            return self._sample_uniform()
