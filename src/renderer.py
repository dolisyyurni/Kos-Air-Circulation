"""PyVista renderer for the building, selected-floor CFD field, and particles."""

import time

import numpy as np
import pyvista as pv

from src import building
from src.airflow import AirflowField, sample_velocity


class Renderer:
    COLORS = ["#35D07F", "#FFD43B", "#FF4D4D"]

    def __init__(self, config: dict, state, field: AirflowField, tracker=None, gestures=None):
        self.config = config
        self.state = state
        self.field = field
        self.tracker = tracker
        self.gestures = gestures
        self.plotter = building.plotter
        self.rng = np.random.default_rng(7)
        self.particle_count = int(config["simulation"]["particle_count"])
        self.particles = self._spawn_particles(self.particle_count)
        self.active_floor = -1
        self._last_time = time.perf_counter()
        self._field_mesh = None
        self._particle_mesh = pv.PolyData(self._world_points(self.particles))
        self._setup()

    def _world_points(self, xy):
        z = self.state.floor * building.FLOOR_HEIGHT + 0.22
        return np.column_stack((xy, np.full(len(xy), z)))

    def _spawn_particles(self, count):
        candidates = np.argwhere(~self.field.solid)
        if not len(candidates):
            return np.zeros((count, 2))
        chosen = candidates[self.rng.integers(0, len(candidates), count)]
        spacing = self.field.x[1] - self.field.x[0]
        return np.column_stack((self.field.x[chosen[:, 1]], self.field.y[chosen[:, 0]])) + self.rng.normal(0, spacing * 0.15, (count, 2))

    def _bands(self, speed):
        slow = float(self.config["visualization"]["slow_speed"])
        fast = float(self.config["visualization"]["fast_speed"])
        return np.where(speed < slow, 0, np.where(speed < fast, 1, 2))

    def _setup(self):
        self.plotter.add_text(
            "CFD 2D | HIJAU: LAMBAT  KUNING: SEDANG  MERAH: CEPAT",
            position="upper_right", color="white", font_size=12,
        )
        self.floor_text = None
        self._update_floor_mesh(force=True)
        initial_speed = np.linalg.norm(sample_velocity(self.field, self.particles), axis=1)
        self._particle_mesh.point_data["speed_band"] = self._bands(initial_speed)
        self.particle_actor = self.plotter.add_mesh(
            self._particle_mesh, scalars="speed_band", cmap=self.COLORS, clim=(0, 2),
            n_colors=3, render_points_as_spheres=True, point_size=8,
            show_scalar_bar=False, name="air-particles",
        )
        self.particle_actor.SetVisibility(bool(self.config["visualization"].get("show_particles", True)))
        for key in "123":
            self.plotter.add_key_event(key, lambda floor=int(key) - 1: self.state.select_floor(floor))
        self.plotter.add_key_event("space", self.toggle_running)
        self.plotter.add_key_event("q", self.plotter.close)

    def _update_floor_mesh(self, force=False):
        if self.active_floor == self.state.floor and not force:
            return
        rows, cols = np.where(~self.field.solid)
        xy = np.column_stack((self.field.x[cols], self.field.y[rows]))
        points = self._world_points(xy)
        speeds = self.field.speed[rows, cols]
        if self._field_mesh is None:
            self._field_mesh = pv.PolyData(points)
            self._field_mesh.point_data["speed_band"] = self._bands(speeds)
            self.plotter.add_mesh(
                self._field_mesh, scalars="speed_band", cmap=self.COLORS, clim=(0, 2),
                n_colors=3, render_points_as_spheres=True, point_size=5,
                show_scalar_bar=False, name="cfd-velocity-field",
            )
        else:
            self._field_mesh.points = points
        self._particle_mesh.points = self._world_points(self.particles)
        if self.floor_text is not None:
            self.plotter.remove_actor(self.floor_text, render=False)
        self.floor_text = self.plotter.add_text(
            f"LANTAI {self.state.floor + 1}/{self.state.floor_count}  |  PINCH: JEDA/LANJUT",
            position="lower_left", color="white", font_size=13,
        )
        self.active_floor = self.state.floor

    def toggle_running(self):
        self.state.running = not self.state.running

    def _apply_gestures(self):
        if self.gestures is None or self.tracker is None:
            return
        for event, value in self.gestures.update(self.tracker.landmarks).items():
            if event == "floor":
                self.state.select_floor(value)
            elif event == "toggle_running":
                self.toggle_running()

    def _step_particles(self, dt):
        if not self.state.running:
            return
        scale = float(self.config["simulation"]["particle_speed_scale"])
        velocity = sample_velocity(self.field, self.particles)
        self.particles += velocity * (dt * scale)
        spacing = self.field.x[1] - self.field.x[0]
        ix = np.clip((self.particles[:, 0] / spacing).astype(int), 0, len(self.field.x) - 1)
        iy = np.clip((self.particles[:, 1] / spacing).astype(int), 0, len(self.field.y) - 1)
        invalid = ((self.particles[:, 0] < 0) | (self.particles[:, 0] > building.WIDTH) |
                   (self.particles[:, 1] < 0) | (self.particles[:, 1] > building.DEPTH) |
                   self.field.solid[iy, ix])
        if invalid.any():
            self.particles[invalid] = self._spawn_particles(int(invalid.sum()))
        self._particle_mesh.points = self._world_points(self.particles)
        speeds = np.linalg.norm(sample_velocity(self.field, self.particles), axis=1)
        self._particle_mesh.point_data["speed_band"] = self._bands(speeds)

    def _on_timer(self, _step):
        """Advance the animation from VTK's UI event loop."""
        now = time.perf_counter()
        dt = min(now - self._last_time, 0.08)
        self._last_time = now
        self._apply_gestures()
        if self.tracker is not None:
            self.tracker.display()
        self._update_floor_mesh()
        self._step_particles(dt)
        if self.tracker is not None and self.tracker.error:
            print(f"Hand tracking dihentikan: {self.tracker.error}")
            self.tracker.close()
            self.tracker = None
            self.gestures = None

    def run(self):
        """Run the blocking UI loop until the user closes the 3D window."""
        try:
            self.plotter.add_timer_event(
                max_steps=2**31 - 1,
                duration=16,
                callback=self._on_timer,
            )
            # interactive=True enters VTK's native event loop; the previous
            # non-blocking loop could see GetDone() already set and exit at once.
            self.plotter.show(interactive=True, auto_close=False)
        finally:
            if self.tracker is not None:
                self.tracker.close()
            self.plotter.close()
