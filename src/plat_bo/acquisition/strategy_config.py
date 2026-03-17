from __future__ import annotations

from typing import Any

COMPONENT_KEYS = (
    "surrogate_model",
    "acquisition_function",
    "inner_optimizer",
    "hyperparameter_update",
)

SUPPORTED_SURROGATE_MODELS = {
    "single_task_gp",
    "turbo_trust_region_skeleton",
}
SUPPORTED_ACQUISITION_FUNCTIONS = {
    "expected_improvement",
    "upper_confidence_bound",
    "local_candidate_ranking",
}
SUPPORTED_INNER_OPTIMIZERS = {
    "botorch_optimize_acqf",
    "candidate_pool_nearest_distance",
}
SUPPORTED_HYPERPARAMETER_UPDATES = {
    "mll_fit",
    "turbo_state_update_reserved",
}

COMPONENT_COMPATIBILITY_MATRIX = [
    {
        "surrogate_model": "single_task_gp",
        "acquisition_function": "expected_improvement",
        "inner_optimizer": "botorch_optimize_acqf",
        "hyperparameter_update": "mll_fit",
    },
    {
        "surrogate_model": "single_task_gp",
        "acquisition_function": "upper_confidence_bound",
        "inner_optimizer": "botorch_optimize_acqf",
        "hyperparameter_update": "mll_fit",
    },
    {
        "surrogate_model": "turbo_trust_region_skeleton",
        "acquisition_function": "local_candidate_ranking",
        "inner_optimizer": "candidate_pool_nearest_distance",
        "hyperparameter_update": "turbo_state_update_reserved",
    },
]

STRATEGY_PRESETS = {
    "base_single_task_gp_ei": {
        "surrogate_model": "single_task_gp",
        "acquisition_function": "expected_improvement",
        "inner_optimizer": "botorch_optimize_acqf",
        "hyperparameter_update": "mll_fit",
    },
    "base_single_task_gp_ucb": {
        "surrogate_model": "single_task_gp",
        "acquisition_function": "upper_confidence_bound",
        "inner_optimizer": "botorch_optimize_acqf",
        "hyperparameter_update": "mll_fit",
    },
    "base_turbo_gp_ei_skeleton": {
        "surrogate_model": "turbo_trust_region_skeleton",
        "acquisition_function": "local_candidate_ranking",
        "inner_optimizer": "candidate_pool_nearest_distance",
        "hyperparameter_update": "turbo_state_update_reserved",
    },
}
SUPPORTED_STRATEGIES = set(STRATEGY_PRESETS.keys())

STRATEGY_COMPONENT_GUIDE = {
    "base_single_task_gp_ei": {
        **STRATEGY_PRESETS["base_single_task_gp_ei"],
        "notes": "默认低维连续优化基线策略",
    },
    "base_single_task_gp_ucb": {
        **STRATEGY_PRESETS["base_single_task_gp_ucb"],
        "notes": "探索更积极，适合早期探索阶段",
    },
    "base_turbo_gp_ei_skeleton": {
        **STRATEGY_PRESETS["base_turbo_gp_ei_skeleton"],
        "notes": "高维入口骨架，后续升级完整 TuRBO 状态机",
    },
}


def normalize_strategy_name(strategy: str) -> str:
    return strategy.strip().lower()


def _normalize_component_config(config: dict[str, Any]) -> dict[str, str]:
    return {
        str(k): str(v).strip().lower()
        for k, v in config.items()
        if isinstance(k, str) and v is not None
    }


def ensure_supported_strategy(strategy: str) -> str:
    normalized = normalize_strategy_name(strategy)
    if normalized not in SUPPORTED_STRATEGIES:
        raise ValueError(
            "unsupported strategy '{}', available={}".format(
                strategy, sorted(SUPPORTED_STRATEGIES)
            )
        )
    return normalized


def ensure_component_config_supported(component_config: dict[str, Any]) -> dict[str, str]:
    normalized = _normalize_component_config(component_config)
    missing = [k for k in COMPONENT_KEYS if k not in normalized]
    if missing:
        raise ValueError("component_config missing fields: {}".format(missing))
    extra = [k for k in normalized.keys() if k not in COMPONENT_KEYS]
    if extra:
        raise ValueError("component_config has unknown fields: {}".format(extra))

    if normalized["surrogate_model"] not in SUPPORTED_SURROGATE_MODELS:
        raise ValueError(
            "unsupported surrogate_model '{}', available={}".format(
                normalized["surrogate_model"], sorted(SUPPORTED_SURROGATE_MODELS)
            )
        )
    if normalized["acquisition_function"] not in SUPPORTED_ACQUISITION_FUNCTIONS:
        raise ValueError(
            "unsupported acquisition_function '{}', available={}".format(
                normalized["acquisition_function"], sorted(SUPPORTED_ACQUISITION_FUNCTIONS)
            )
        )
    if normalized["inner_optimizer"] not in SUPPORTED_INNER_OPTIMIZERS:
        raise ValueError(
            "unsupported inner_optimizer '{}', available={}".format(
                normalized["inner_optimizer"], sorted(SUPPORTED_INNER_OPTIMIZERS)
            )
        )
    if normalized["hyperparameter_update"] not in SUPPORTED_HYPERPARAMETER_UPDATES:
        raise ValueError(
            "unsupported hyperparameter_update '{}', available={}".format(
                normalized["hyperparameter_update"], sorted(SUPPORTED_HYPERPARAMETER_UPDATES)
            )
        )

    if normalized not in COMPONENT_COMPATIBILITY_MATRIX:
        raise ValueError(
            "incompatible component combination: {}".format(
                {k: normalized[k] for k in COMPONENT_KEYS}
            )
        )
    return normalized


def infer_strategy_from_component_config(component_config: dict[str, Any]) -> str:
    normalized = ensure_component_config_supported(component_config)
    for strategy, preset in STRATEGY_PRESETS.items():
        if normalized == preset:
            return strategy
    return "component_combo_custom"


def resolve_component_config(
    strategy: str | None, component_config: dict[str, Any] | None
) -> tuple[str, dict[str, str]]:
    merged: dict[str, Any] = {}
    normalized_strategy = ""
    if strategy is not None and str(strategy).strip():
        normalized_strategy = ensure_supported_strategy(str(strategy))
        merged.update(STRATEGY_PRESETS[normalized_strategy])
    if component_config is not None:
        merged.update(_normalize_component_config(component_config))
    if not merged:
        raise ValueError("either strategy or component_config must be provided")
    normalized_components = ensure_component_config_supported(merged)
    if not normalized_strategy:
        normalized_strategy = infer_strategy_from_component_config(normalized_components)
    return normalized_strategy, normalized_components
