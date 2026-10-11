"""Unit tests for the coarse-grid airflow solver and particle sampling."""

import unittest

import numpy as np

from src.airflow import _wall_mask, sample_velocity, solve_floor
from src.settings import DEFAULT_CONFIG


class AirflowTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            **DEFAULT_CONFIG,
            "building": dict(DEFAULT_CONFIG["building"]),
            "simulation": {
                **DEFAULT_CONFIG["simulation"],
                "grid_size": 24,
                "solver_steps": 8,
                "pressure_iterations": 8,
                "time_step": 0.008,
            },
        }

    def test_solver_returns_finite_grid_matching_config(self):
        field = solve_floor(self.config)

        self.assertEqual(field.u.shape, (24, 24))
        self.assertEqual(field.v.shape, (24, 24))
        self.assertEqual(field.solid.shape, (24, 24))
        self.assertTrue(np.isfinite(field.u).all())
        self.assertTrue(np.isfinite(field.v).all())
        self.assertGreater(float(field.speed.max()), 0.0)

    def test_solid_cells_have_zero_velocity(self):
        field = solve_floor(self.config)

        self.assertTrue(np.all(field.u[field.solid] == 0.0))
        self.assertTrue(np.all(field.v[field.solid] == 0.0))

    def test_wall_mask_blocks_courtyard_and_outer_boundary(self):
        x, y, solid = _wall_mask(24)
        center_x = int(np.abs(x - 9.0).argmin())
        center_y = int(np.abs(y - 7.2).argmin())

        self.assertTrue(solid[center_y, center_x])
        self.assertTrue(solid[0, :].all())
        self.assertTrue(solid[-1, :].all())

    def test_particle_velocity_sampling_returns_xy_vectors(self):
        field = solve_floor(self.config)
        positions = np.array([[3.0, 3.0], [8.0, 8.0], [15.0, 12.0]])

        velocity = sample_velocity(field, positions)

        self.assertEqual(velocity.shape, (3, 2))
        self.assertTrue(np.isfinite(velocity).all())


if __name__ == "__main__":
    unittest.main()
