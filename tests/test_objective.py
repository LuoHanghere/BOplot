import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from plat_bo.objective.branin_program import branin

class TestObjective(unittest.TestCase):
    def test_branin_value(self):
        # Branin global minimum at [-pi, 12.275], [pi, 2.275], [9.42478, 2.475]
        # Approx 0.397887
        val = branin(-3.14159, 12.275)
        self.assertAlmostEqual(val, 0.397887, places=4)

if __name__ == '__main__':
    unittest.main()
