"""A lightweight 2-D incompressible airflow solver for one building floor.

This is an educational, coarse-grid Navier-Stokes projection model. It is useful
for interactive visualization; it is not a validated engineering CFD solver.
"""

from dataclasses import dataclass

import numpy as np

from src import building


@dataclass
class AirflowField:
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    solid: np.ndarray

    @property
    def speed(self):
        return np.hypot(self.u, self.v)


def _wall_mask(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize the floor-plan partitions and courtyard, leaving openings."""
    x = np.linspace(0.0, building.WIDTH, size)
    y = np.linspace(0.0, building.DEPTH, size)
    xx, yy = np.meshgrid(x, y)
    solid = np.zeros((size, size), dtype=bool)
    half_wall = min(building.WIDTH, building.DEPTH) / size * 0.55

    def segment(x1, y1, x2, y2, openings=()):
        if x1 == x2:
            hit = (np.abs(xx - x1) <= half_wall) & (yy >= min(y1, y2)) & (yy <= max(y1, y2))
            for center, width in openings:
                hit &= ~((np.abs(yy - center) <= width / 2) & (np.abs(xx - x1) <= half_wall * 2))
        else:
            hit = (np.abs(yy - y1) <= half_wall) & (xx >= min(x1, x2)) & (xx <= max(x1, x2))
            for center, width in openings:
                hit &= ~((np.abs(xx - center) <= width / 2) & (np.abs(yy - y1) <= half_wall * 2))
        solid[:] |= hit

    # Interior partitions from building.py; doors/windows are represented as gaps.
    for a, b in [(0, 6), (6, 7), (11, 12), (12, 18)]:
        segment(a, 10, b, 10, openings=((2.5, 1.0), (6.45, 0.9), (8.5, 1.0), (11.45, 0.9), (14.5, 1.0)))
    for a, b in [(0, 2), (2.9, 6), (6.9, 7), (11, 12), (12.9, 13), (13.9, 18)]:
        segment(a, 4.5, b, 4.5, openings=((3.5, 1.0), (6.45, 0.9), (8.5, 1.0), (12.45, 0.9), (15.0, 1.0)))
    for xv, ranges in [(6, [(0, 4.5), (10, 18)]), (12, [(0, 4.5), (10, 18)]), (13, [(0, 4.5)])]:
        for ya, yb in ranges:
            segment(xv, ya, xv, yb)

    # Courtyard is an open-to-sky void, not a walkable floor cell.
    c = building.COURTYARD
    solid |= (xx >= c["x1"]) & (xx <= c["x2"]) & (yy >= c["y1"]) & (yy <= c["y2"])
    # Closed exterior boundaries; windows and the entrance are modeled as air sources.
    solid[[0, -1], :] = True
    solid[:, [0, -1]] = True
    return x, y, solid


def solve_floor(config: dict) -> AirflowField:
    """Compute a steady-looking 2-D flow field using a pressure projection."""
    size = int(config["simulation"]["grid_size"])
    x, y, solid = _wall_mask(size)
    h = x[1] - x[0]
    dt = float(config["simulation"]["time_step"])
    viscosity = float(config["simulation"]["viscosity"])
    inlet_speed = float(config["simulation"]["inlet_speed"])
    steps = int(config["simulation"]["solver_steps"])
    pressure_steps = int(config["simulation"]["pressure_iterations"])
    xx, yy = np.meshgrid(x, y)
    u = np.zeros_like(xx)
    v = np.zeros_like(yy)
    p = np.zeros_like(xx)

    # Distributed window/door forcing based on the openings in the floor plan.
    fx = np.zeros_like(xx)
    fy = np.zeros_like(yy)
    for wx, wy, dx, dy in [(3.5, 4.5, 1, 0), (8.5, 4.5, 0.7, 0.2),
                           (15, 4.5, -1, 0), (2.5, 10, 1, 0),
                           (8.5, 10, -0.5, -0.2), (14.5, 10, -1, 0)]:
        blob = np.exp(-((xx - wx) ** 2 + (yy - wy) ** 2) / 1.3)
        fx += inlet_speed * dx * blob * 2.0
        fy += inlet_speed * dy * blob * 2.0

    fluid = ~solid
    for _ in range(steps):
        dudx = (np.roll(u, -1, 1) - np.roll(u, 1, 1)) / (2 * h)
        dudy = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2 * h)
        dvdx = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * h)
        dvdy = (np.roll(v, -1, 0) - np.roll(v, 1, 0)) / (2 * h)
        lap_u = (np.roll(u, 1, 1) + np.roll(u, -1, 1) + np.roll(u, 1, 0) + np.roll(u, -1, 0) - 4 * u) / h**2
        lap_v = (np.roll(v, 1, 1) + np.roll(v, -1, 1) + np.roll(v, 1, 0) + np.roll(v, -1, 0) - 4 * v) / h**2
        u_star = u + dt * (-(u * dudx + v * dudy) + viscosity * lap_u + fx)
        v_star = v + dt * (-(u * dvdx + v * dvdy) + viscosity * lap_v + fy)
        u_star[solid] = 0
        v_star[solid] = 0
        divergence = ((np.roll(u_star, -1, 1) - np.roll(u_star, 1, 1)) +
                      (np.roll(v_star, -1, 0) - np.roll(v_star, 1, 0))) / (2 * h)
        for _ in range(pressure_steps):
            p_next = (np.roll(p, 1, 1) + np.roll(p, -1, 1) +
                      np.roll(p, 1, 0) + np.roll(p, -1, 0) - divergence * h**2 / dt) / 4
            p_next[solid] = 0
            p = p_next
        u = u_star - dt * (np.roll(p, -1, 1) - np.roll(p, 1, 1)) / (2 * h)
        v = v_star - dt * (np.roll(p, -1, 0) - np.roll(p, 1, 0)) / (2 * h)
        u[solid] = 0
        v[solid] = 0
        # Keep the explicit scheme bounded for interactive use.
        u = np.clip(u, -inlet_speed * 2.5, inlet_speed * 2.5)
        v = np.clip(v, -inlet_speed * 2.5, inlet_speed * 2.5)

    return AirflowField(x=x, y=y, u=u, v=v, solid=solid)


def sample_velocity(field: AirflowField, positions: np.ndarray) -> np.ndarray:
    """Bilinearly sample the grid velocity at world-space particle positions."""
    positions = np.asarray(positions, dtype=float)
    spacing = field.x[1] - field.x[0]
    ix = np.clip((positions[:, 0] / spacing).astype(int), 0, len(field.x) - 2)
    iy = np.clip((positions[:, 1] / spacing).astype(int), 0, len(field.y) - 2)
    tx = np.clip(positions[:, 0] / spacing - ix, 0, 1)
    ty = np.clip(positions[:, 1] / spacing - iy, 0, 1)
    def interp(values):
        return ((1 - tx) * (1 - ty) * values[iy, ix] + tx * (1 - ty) * values[iy, ix + 1] +
                (1 - tx) * ty * values[iy + 1, ix] + tx * ty * values[iy + 1, ix + 1])
    return np.column_stack((interp(field.u), interp(field.v)))
