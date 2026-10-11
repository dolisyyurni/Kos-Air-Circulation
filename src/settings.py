"""Application configuration and shared visualization state."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = {
    "building": {"width": 18.0, "depth": 18.0, "floor_height": 3.2, "floors": 3},
    "simulation": {
        "grid_size": 56, "time_step": 0.012, "viscosity": 0.045,
        "inlet_speed": 0.9, "solver_steps": 55, "pressure_iterations": 24,
        "particle_count": 700, "particle_speed_scale": 0.8,
    },
    "visualization": {
        "window_size": [1400, 900], "background": "black", "wall_color": "white",
        "slow_speed": 0.25, "fast_speed": 0.65, "show_particles": True,
    },
    "hand_tracking": {"enabled": True, "camera_index": 0, "max_hands": 1},
}


def load_config(path: str | Path | None = None) -> dict:
    """Load JSON settings, filling any omitted sections from safe defaults."""
    config_path = Path(path) if path else ROOT / "config.json"
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path.exists() and config_path.stat().st_size:
        with config_path.open("r", encoding="utf-8") as source:
            supplied = json.load(source)
        for section, values in supplied.items():
            if isinstance(values, dict) and section in config:
                config[section].update(values)
            else:
                config[section] = values
    return config


class AppState:
    def __init__(self, floors: int = 3):
        self.floor = 0
        self.floor_count = floors
        self.running = True
        self.camera_yaw = 0.0

    def select_floor(self, floor: int):
        self.floor = max(0, min(self.floor_count - 1, int(floor)))
