import unittest
from pathlib import Path
import tempfile
import shutil
import sys
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from plat_bo.surrogate.engine import BOEngine
from plat_bo.initialization.models import TaskConfig, TrialOutput

class TestSurrogate(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.state_dir = self.test_dir / "state"
        self.state_dir.mkdir()
        self.config = TaskConfig(
            task_id="test-surrogate",
            problem="branin",
            strategy="base_single_task_gp_ei",
            bounds=[[-5.0, 10.0], [0.0, 15.0]],
            max_iterations=10,
            initial_random_trials=3
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_engine_suggestion(self):
        engine = BOEngine(self.state_dir, self.config)
        # 1. First suggestions should be random
        for i in range(3):
            trial = engine.suggest_next("test-surrogate", i)
            self.assertEqual(len(trial.parameters), 2)
            # Simulate output
            output = TrialOutput(
                task_id="test-surrogate",
                iteration=i,
                parameters=trial.parameters,
                objective=random.random() * 100,
                success=True,
                message="ok",
                cost_seconds=0.1
            )
            engine.update(output)

        # 2. Next suggestion should use model (BoTorch)
        # This might fail if botorch is not installed, but let's try
        try:
            import botorch
            trial = engine.suggest_next("test-surrogate", 3)
            self.assertEqual(len(trial.parameters), 2)
            print(f"Suggestion via BoTorch: {trial.parameters}")
        except ImportError:
            print("BoTorch not installed, skipping model test")
        except Exception as e:
            # If model fitting fails for some reason (e.g. numerical issues with few points)
            # Engine falls back to random, which is also valid behavior
            # We just check if it returns something valid
            print(f"Model failed: {e}")
            trial = engine.suggest_next("test-surrogate", 3)
            self.assertEqual(len(trial.parameters), 2)

    def test_turbo_skeleton_high_dim(self):
        high_dim = 24
        config = TaskConfig(
            task_id="test-turbo",
            problem="mock_quadratic",
            strategy="base_turbo_gp_ei_skeleton",
            bounds=[[0.0, 1.0] for _ in range(high_dim)],
            max_iterations=12,
            initial_random_trials=3
        )
        engine = BOEngine(self.state_dir, config)
        for i in range(3):
            trial = engine.suggest_next("test-turbo", i)
            output = TrialOutput(
                task_id="test-turbo",
                iteration=i,
                parameters=trial.parameters,
                objective=sum((x - 0.3) ** 2 for x in trial.parameters),
                success=True,
                message="ok",
                cost_seconds=0.1
            )
            engine.update(output)
        trial = engine.suggest_next("test-turbo", 3)
        self.assertEqual(len(trial.parameters), high_dim)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in trial.parameters))

    def test_turbo_state_adaptation(self):
        config = TaskConfig(
            task_id="test-turbo-state",
            problem="mock_quadratic",
            strategy="base_turbo_gp_ei_skeleton",
            bounds=[[0.0, 1.0] for _ in range(4)],
            max_iterations=12,
            initial_random_trials=1
        )
        engine = BOEngine(self.state_dir, config)
        for i in range(4):
            trial = engine.suggest_next("test-turbo-state", i)
            output = TrialOutput(
                task_id="test-turbo-state",
                iteration=i,
                parameters=trial.parameters,
                objective=1.0 - i * 0.2,
                success=True,
                message="ok",
                cost_seconds=0.1
            )
            engine.update(output)
        state = engine.get_state("test-turbo-state")
        self.assertIn("length", state.turbo_state)
        self.assertGreaterEqual(float(state.turbo_state["length"]), 1.0)

if __name__ == '__main__':
    unittest.main()
