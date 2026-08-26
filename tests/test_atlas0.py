import unittest
import numpy as np

from matrix_atlas import (
    dense_signed_basis,
    PositiveGrowthRow,
    best_reachable_row,
    run_task,
)


class Atlas0Tests(unittest.TestCase):
    def test_mass_is_conserved_and_nonnegative(self):
        R = dense_signed_basis(0)
        target = 0.4 * np.eye(6)[0]
        learner = PositiveGrowthRow(R).fit(target, steps=200)
        self.assertAlmostEqual(float(learner.mass.sum()), 1.0, places=10)
        self.assertGreaterEqual(float(learner.mass.min()), 0.001 - 1e-12)

    def test_effective_row_can_be_signed(self):
        R = dense_signed_basis(1)
        learner = PositiveGrowthRow(R)
        w = learner.effective_row
        self.assertTrue(np.any(w < 0.0) or np.any(w > 0.0))

    def test_hull_attacker_solves_reachable_row(self):
        R = dense_signed_basis(2)
        reserve = 0.001
        mass = np.full(len(R), reserve)
        free = 1.0 - len(R) * reserve
        mass[[1, 7, 11, 20]] += free * np.array([0.1, 0.2, 0.3, 0.4])
        target = mass @ R
        _, hull = best_reachable_row(R, target, reserve=reserve)
        self.assertLess(float(np.linalg.norm(hull - target)), 2e-3)

    def test_reachable_matrix_is_learnable(self):
        r = run_task("reachable_sparse", seed=0)
        self.assertLess(r["target_relative_fro_error"], 0.15)
        self.assertLess(r["heldout_nmse"], 0.03)

    def test_low_rank_target_stays_low_rank(self):
        r = run_task("low_rank", seed=0)
        self.assertEqual(r["target_rank"], 2)
        self.assertLessEqual(r["grown_rank"], 4)
        self.assertLess(r["target_relative_fro_error"], 0.03)

    def test_selector_exposes_representation_gap(self):
        r = run_task("selector", seed=0)
        self.assertGreater(r["hull_relative_fro_error"], 0.02)
        self.assertGreaterEqual(r["target_relative_fro_error"], r["hull_relative_fro_error"] - 1e-6)


if __name__ == "__main__":
    unittest.main()
