from __future__ import annotations
from typing import TYPE_CHECKING

from .strategy_config import resolve_component_config

if TYPE_CHECKING:
    from ..initialization.models import TaskConfig

def suggest_botorch(
    x_train: list[list[float]],
    y_train: list[float],
    task_config: TaskConfig,
    components: dict[str, str] | None = None,
) -> list[float]:
    import torch
    from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
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
    bounds = torch.tensor(task_config.bounds, dtype=dtype, device=device)
    x = torch.tensor(x_train, dtype=dtype, device=device)
    y = torch.tensor(y_train, dtype=dtype, device=device).unsqueeze(-1)

    lower = bounds[:, 0]
    upper = bounds[:, 1]
    x_norm = (x - lower) / (upper - lower)
    x_norm = x_norm.clamp(0.0, 1.0)

    if task_config.objective_direction == "minimize":
        y_model = -y
    else:
        y_model = y

    resolved_components = components
    if resolved_components is None:
        _, resolved_components = resolve_component_config(
            task_config.strategy, task_config.component_config
        )
    if resolved_components["surrogate_model"] != "single_task_gp":
        raise ValueError(
            "suggest_botorch requires surrogate_model='single_task_gp', got {}".format(
                resolved_components["surrogate_model"]
            )
        )

    model = SingleTaskGP(
        train_X=x_norm,
        train_Y=y_model,
        input_transform=Normalize(d=task_config.dim),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if resolved_components["hyperparameter_update"] == "mll_fit":
        _fit_gp(mll)
    else:
        raise ValueError(
            "unsupported hyperparameter_update for single_task_gp: {}".format(
                resolved_components["hyperparameter_update"]
            )
        )

    if resolved_components["acquisition_function"] == "upper_confidence_bound":
        acq = UpperConfidenceBound(model=model, beta=0.1)
    elif resolved_components["acquisition_function"] == "expected_improvement":
        best_f = y_model.max().item()
        acq = ExpectedImprovement(model=model, best_f=best_f)
    else:
        raise ValueError(
            "unsupported acquisition_function for single_task_gp: {}".format(
                resolved_components["acquisition_function"]
            )
        )
    unit_bounds = torch.stack(
        [
            torch.zeros(task_config.dim, dtype=dtype, device=device),
            torch.ones(task_config.dim, dtype=dtype, device=device),
        ]
    )

    if resolved_components["inner_optimizer"] == "botorch_optimize_acqf":
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=unit_bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
            options={"batch_limit": 5, "maxiter": 100},
        )
    else:
        raise ValueError(
            "unsupported inner_optimizer for single_task_gp: {}".format(
                resolved_components["inner_optimizer"]
            )
        )
    x_next_norm = candidate.detach().cpu().squeeze(0)
    x_next = lower.cpu() + x_next_norm * (upper.cpu() - lower.cpu())
    return [round(float(v.item()), 6) for v in x_next]
