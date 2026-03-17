import unittest
from pathlib import Path
import tempfile
import shutil
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from plat_bo.initialization.models import TaskConfig, TrialInput
from plat_bo.initialization.task_config_store import save_task_config, load_task_config
from plat_bo.initialization.validator import ValidationError, validate_task_config

class TestInitialization(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_dir = self.test_dir / "configs"
        self.config_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_task_config_io(self):
        config = TaskConfig(
            task_id="test-001",
            problem="branin",
            strategy="base_single_task_gp_ei",
            bounds=[[-5.0, 10.0], [0.0, 15.0]],
            max_iterations=10,
            initial_random_trials=5
        )
        save_task_config(self.config_dir, config)
        loaded = load_task_config(self.config_dir, "test-001")
        self.assertEqual(loaded.task_id, "test-001")
        self.assertEqual(loaded.bounds, [[-5.0, 10.0], [0.0, 15.0]])

    def test_strategy_validation(self):
        valid = {
            "task_id": "test-002",
            "problem": "branin",
            "strategy": "base_single_task_gp_ucb",
            "bounds": [[-5.0, 10.0], [0.0, 15.0]],
            "max_iterations": 10,
        }
        validate_task_config(valid)
        invalid = dict(valid)
        invalid["strategy"] = "unknown_strategy"
        with self.assertRaises(ValidationError):
            validate_task_config(invalid)
        turbo = dict(valid)
        turbo["strategy"] = "base_turbo_gp_ei_skeleton"
        validate_task_config(turbo)
        component_only = {
            "task_id": "test-003",
            "problem": "branin",
            "component_config": {
                "surrogate_model": "single_task_gp",
                "acquisition_function": "expected_improvement",
                "inner_optimizer": "botorch_optimize_acqf",
                "hyperparameter_update": "mll_fit",
            },
            "bounds": [[-5.0, 10.0], [0.0, 15.0]],
            "max_iterations": 10,
        }
        validate_task_config(component_only)

if __name__ == '__main__':
    unittest.main()
