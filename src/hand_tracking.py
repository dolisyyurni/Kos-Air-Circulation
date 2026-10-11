"""Optional webcam hand tracking using MediaPipe Tasks Hand Landmarker."""

import tempfile
import threading
import time
from pathlib import Path
from urllib.request import urlretrieve


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

class HandTracker:
    def __init__(self, camera_index: int = 0, max_hands: int = 1, model_path=None):
        self.camera_index = camera_index
        self.max_hands = max_hands
        self.landmarks = None
        self.frame = None
        self.error = None
        self._stop = threading.Event()
        self._thread = None
        self._camera = None
        self._cv2 = None
        self._mediapipe = None
        self.model_path = Path(model_path) if model_path else None

    def start(self):
        try:
            import cv2
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError("OpenCV/MediaPipe belum terpasang.") from error
        self._cv2 = cv2
        self._mediapipe = mp
        self.model_path = self._ensure_model()
        self._camera = cv2.VideoCapture(self.camera_index)
        if not self._camera.isOpened():
            self._camera.release()
            self._camera = None
            raise RuntimeError(f"Tidak dapat membuka webcam index {self.camera_index}.")
        self._thread = threading.Thread(target=self._run, name="hand-tracking", daemon=True)
        self._thread.start()

    def _ensure_model(self):
        if self.model_path is not None and self.model_path.is_file():
            return self.model_path
        project_model = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
        if project_model.is_file():
            return project_model
        cache_dir = Path(tempfile.gettempdir()) / "KosAirCirculation"
        cache_model = cache_dir / "hand_landmarker.task"
        if cache_model.is_file() and cache_model.stat().st_size > 0:
            return cache_model

        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_model = cache_model.with_suffix(".download")
        try:
            urlretrieve(MODEL_URL, temporary_model)
            temporary_model.replace(cache_model)
        except Exception as error:
            temporary_model.unlink(missing_ok=True)
            raise RuntimeError(
                "Model tangan MediaPipe belum tersedia. Periksa koneksi internet "
                "atau set hand_tracking.model_path di config.json."
            ) from error
        return cache_model

    def _run(self):
        try:
            mp = self._mediapipe
            vision = mp.tasks.vision
            options = vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.max_hands,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            with vision.HandLandmarker.create_from_options(options) as landmarker:
                while not self._stop.is_set():
                    ok, frame = self._camera.read()
                    if not ok:
                        self.landmarks = None
                        continue
                    frame = self._cv2.flip(frame, 1)
                    rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = time.monotonic_ns() // 1_000_000
                    result = landmarker.detect_for_video(image, timestamp_ms)
                    if result.hand_landmarks:
                        hand = result.hand_landmarks[0]
                        self.landmarks = [(point.x, point.y, point.z) for point in hand]
                        self._draw_landmarks(frame, self.landmarks)
                    else:
                        self.landmarks = None
                    self.frame = frame
        except Exception as error:  # Keep rendering alive if camera tracking fails.
            self.error = error
            self.landmarks = None
            self.frame = None

    def _draw_landmarks(self, frame, landmarks):
        height, width = frame.shape[:2]
        points = [(int(x * width), int(y * height)) for x, y, _ in landmarks]
        for start, end in HAND_CONNECTIONS:
            self._cv2.line(frame, points[start], points[end], (70, 220, 100), 2)
        for point in points:
            self._cv2.circle(frame, point, 3, (0, 90, 255), -1)

    def display(self):
        """Show the latest mirrored webcam image; call from the UI thread."""
        if self.frame is not None:
            self._cv2.imshow("Webcam | kontrol tangan", self.frame)
            self._cv2.waitKey(1)

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)
        if self._camera:
            self._camera.release()
            self._camera = None
        if self._cv2 is not None:
            try:
                self._cv2.destroyWindow("Webcam | kontrol tangan")
            except self._cv2.error:
                # The webcam window may not have been displayed yet.
                pass
