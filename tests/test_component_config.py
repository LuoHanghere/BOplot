import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from plat_bo.acquisition.strategy_config import resolve_component_config


class TestComponentConfig(unittest.TestCase):
    def test_resolve_from_strategy(self):
        strategy, components = resolve_component_config("base_single_task_gp_ei", None)
        self.assertEqual(strategy, "base_single_task_gp_ei")
        self.assertEqual(components["surrogate_model"], "single_task_gp")
        self.assertEqual(components["acquisition_function"], "expected_improvement")

    def test_resolve_from_components_only(self):
        strategy, components = resolve_component_config(
            None,
            {
                "surrogate_model": "single_task_gp",
                "acquisition_function": "upper_confidence_bound",
                "inner_optimizer": "botorch_optimize_acqf",
                "hyperparameter_update": "mll_fit",
            },
        )
        self.assertEqual(strategy, "base_single_task_gp_ucb")
        self.assertEqual(components["inner_optimizer"], "botorch_optimize_acqf")

    def test_incompatible_components_raise(self):
        with self.assertRaises(ValueError):
            resolve_component_config(
                None,
                {
                    "surrogate_model": "single_task_gp",
                    "acquisition_function": "local_candidate_ranking",
                    "inner_optimizer": "botorch_optimize_acqf",
                    "hyperparameter_update": "mll_fit",
                },
            )


if __name__ == "__main__":
    unittest.main()
