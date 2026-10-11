"""Application entry point: load settings, solve airflow, track hands, render."""

from src.airflow import solve_floor
from src.gesture_controller import GestureController
from src.hand_tracking import HandTracker
from src.renderer import Renderer
from src.settings import AppState, load_config
from src import building


def main():
    config = load_config()
    state = AppState(floors=building.FLOORS)
    tracker = None
    gestures = None

    if config["hand_tracking"].get("enabled", True):
        try:
            tracker = HandTracker(
                camera_index=int(config["hand_tracking"]["camera_index"]),
                max_hands=int(config["hand_tracking"]["max_hands"]),
                model_path=config["hand_tracking"].get("model_path"),
            )
            tracker.start()
            gestures = GestureController(floor_count=building.FLOORS)
        except (RuntimeError, ImportError, AttributeError) as error:
            print(f"Webcam tidak tersedia; gunakan tombol 1-3 dan spasi. Detail: {error}")
            tracker = None

    try:
        print("Menghitung medan kecepatan CFD 2D...")
        field = solve_floor(config)
        print("Klik jendela visualisasi. Telunjuk: pilih lantai; pinch: jeda/lanjut; tombol 1-3: lantai; spasi: jeda; q: keluar.")
        Renderer(config, state, field, tracker=tracker, gestures=gestures).run()
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    main()
