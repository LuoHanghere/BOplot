from __future__ import annotations

import random

from ..initialization.models import TaskConfig


def _best_index(y_train: list[float], objective_direction: str) -> int:
    if objective_direction == "maximize":
        return max(range(len(y_train)), key=lambda i: y_train[i])
    return min(range(len(y_train)), key=lambda i: y_train[i])


def suggest_turbo_skeleton(
    x_train: list[list[float]],
    y_train: list[float],
    task_config: TaskConfig,
    trust_region_length: float = 1.0,
) -> list[float]:
    if not x_train or not y_train or len(x_train) != len(y_train):
        raise ValueError("training data is required for turbo skeleton suggestion")

    dim = task_config.dim
    idx = _best_index(y_train, task_config.objective_direction)
    center = x_train[idx]
    if len(center) != dim:
        raise ValueError("center dimension mismatch with task_config.bounds")

    base_ratio = max(0.05, min(0.35, 1.8 / (dim ** 0.5)))
    radius_ratio = max(0.03, min(0.45, base_ratio * max(0.1, float(trust_region_length))))
    perturb_prob = min(1.0, 20.0 / max(1, dim))
    candidate_count = max(128, 8 * dim)

    candidates: list[list[float]] = []
    for _ in range(candidate_count):
        changed_any = False
        params: list[float] = []
        for d, (lo, hi) in enumerate(task_config.bounds):
            span = hi - lo
            center_d = float(center[d])
            local_lo = max(lo, center_d - radius_ratio * span)
            local_hi = min(hi, center_d + radius_ratio * span)
            if random.random() < perturb_prob:
                value = local_lo + random.random() * (local_hi - local_lo)
                changed_any = True
            else:
                value = center_d
            params.append(float(value))
        if not changed_any:
            choose_d = random.randrange(dim)
            lo, hi = task_config.bounds[choose_d]
            span = hi - lo
            center_d = float(center[choose_d])
            local_lo = max(lo, center_d - radius_ratio * span)
            local_hi = min(hi, center_d + radius_ratio * span)
            params[choose_d] = float(local_lo + random.random() * (local_hi - local_lo))
        candidates.append(params)

    def distance2(candidate: list[float], point: list[float]) -> float:
        total = 0.0
        for d, (lo, hi) in enumerate(task_config.bounds):
            span = max(1e-12, hi - lo)
            diff = (candidate[d] - point[d]) / span
            total += diff * diff
        return total

    best_candidate = candidates[0]
    best_score = -1.0
    for candidate in candidates:
        nearest = min(distance2(candidate, point) for point in x_train)
        if nearest > best_score:
            best_score = nearest
            best_candidate = candidate

    return [round(x, 6) for x in best_candidate]
