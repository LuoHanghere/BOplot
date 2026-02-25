from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Tuple

from .file_io import read_json, write_json
from .models import TaskConfig, TrialInput, TrialOutput


@dataclass
class EngineState:
    task_id: str
    history: List[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BOEngine:
    def __init__(self, state_dir: Path, task_config: TaskConfig) -> None:
        self.state_file = state_dir / "{}_engine_state.json".format(task_config.task_id)
        self.task_config = task_config

    def _load_or_init(self, task_id: str) -> EngineState:
        if self.state_file.exists():
            data = read_json(self.state_file)
            if data.get("task_id") == task_id:
                return EngineState(task_id=task_id, history=list(data.get("history", [])))
        return EngineState(task_id=task_id, history=[])

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
        self._save(state)

    def suggest_next(self, task_id: str, iteration: int) -> TrialInput:
        state = self._load_or_init(task_id)
        if len(state.history) < self.task_config.initial_random_trials:
            params = self._sample_uniform()
        else:
            params = self._suggest_botorch_or_fallback(state)
        return TrialInput(task_id=task_id, iteration=iteration, parameters=params)

    def _sample_uniform(self) -> list[float]:
        params: list[float] = []
        for lo, hi in self.task_config.bounds:
            params.append(round(lo + (hi - lo) * random.random(), 6))
        return params

    def _extract_train_data(self, state: EngineState) -> Tuple[list[list[float]], list[float]]:
        xs: list[list[float]] = []
        ys: list[float] = []
        for item in state.history:
            if bool(item.get("success")):
                try:
                    xs.append([float(v) for v in item["parameters"]])
                    ys.append(float(item["objective"]))
                except Exception:  # noqa: BLE001
                    continue
        return xs, ys

    def _suggest_botorch_or_fallback(self, state: EngineState) -> list[float]:
        x_train, y_train = self._extract_train_data(state)
        if len(x_train) < max(3, self.task_config.initial_random_trials):
            return self._sample_uniform()
        try:
            return self._suggest_botorch(x_train, y_train)
        except Exception:
            return self._sample_uniform()

    def _suggest_botorch(self, x_train: list[list[float]], y_train: list[float]) -> list[float]:
        import torch
        from botorch.acquisition.analytic import ExpectedImprovement
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        try:
            from botorch.fit import fit_gpytorch_mll as _fit_gp
        except ImportError:
            from botorch.fit import fit_gpytorch_model as _fit_gp

        dtype = torch.double
        device = torch.device("cpu")
        bounds = torch.tensor(self.task_config.bounds, dtype=dtype, device=device)
        x = torch.tensor(x_train, dtype=dtype, device=device)
        y = torch.tensor(y_train, dtype=dtype, device=device).unsqueeze(-1)

        lower = bounds[:, 0]
        upper = bounds[:, 1]
        x_norm = (x - lower) / (upper - lower)
        x_norm = x_norm.clamp(0.0, 1.0)

        if self.task_config.objective_direction == "minimize":
            y_model = -y
        else:
            y_model = y

        model = SingleTaskGP(
            train_X=x_norm,
            train_Y=y_model,
            input_transform=Normalize(d=self.task_config.dim),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        _fit_gp(mll)

        best_f = y_model.max().item()
        acq = ExpectedImprovement(model=model, best_f=best_f)
        unit_bounds = torch.stack(
            [
                torch.zeros(self.task_config.dim, dtype=dtype, device=device),
                torch.ones(self.task_config.dim, dtype=dtype, device=device),
            ]
        )

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=unit_bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
            options={"batch_limit": 5, "maxiter": 100},
        )
        x_next_norm = candidate.detach().cpu().squeeze(0)
        x_next = lower.cpu() + x_next_norm * (upper.cpu() - lower.cpu())
        return [round(float(v.item()), 6) for v in x_next]
