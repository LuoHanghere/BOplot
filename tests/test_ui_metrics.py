import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from plat_bo.optimizer.ui_app import _compute_pareto_exploration, _compute_parameter_sobol_proxy


class TestUIMetrics(unittest.TestCase):
    def test_pareto_exploration_from_objective_cost(self):
        success = [
            {"iteration": 0, "objective": 2.0, "cost_seconds": 1.0, "success": True},
            {"iteration": 1, "objective": 1.8, "cost_seconds": 1.2, "success": True},
            {"iteration": 2, "objective": 1.7, "cost_seconds": 0.9, "success": True},
        ]
        result = _compute_pareto_exploration(success)
        self.assertEqual(result["point_count"], 3)
        self.assertGreaterEqual(len(result["frontier"]), 1)
        self.assertIn("source", result)

    def test_parameter_sobol_proxy(self):
        success = []
        for i in range(12):
            x1 = i / 11.0
            x2 = (11 - i) / 11.0
            success.append(
                {
                    "parameters": [x1, x2],
                    "objective": (x1 - 0.4) ** 2 + 0.1 * x2,
                    "success": True,
                }
            )
        result = _compute_parameter_sobol_proxy(success)
        self.assertEqual(len(result), 2)
        self.assertIn("sobol_proxy", result[0])


if __name__ == "__main__":
    unittest.main()
