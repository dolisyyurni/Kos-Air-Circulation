"""Map hand landmarks to floor selection and simulation pause/resume."""

import math


class GestureController:
    def __init__(self, floor_count: int):
        self.floor_count = floor_count
        self._was_pinched = False
        self._floor_candidate = None
        self._candidate_frames = 0

    def update(self, landmarks):
        """Return events from normalized MediaPipe landmarks, or {} if no hand."""
        if not landmarks:
            self._floor_candidate = None
            self._candidate_frames = 0
            self._was_pinched = False
            return {}

        index_tip = landmarks[8]
        thumb_tip = landmarks[4]
        pinch_distance = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
        pinched = pinch_distance < 0.055
        events = {"toggle_running": pinched and not self._was_pinched}
        self._was_pinched = pinched

        # The camera feed is mirrored: hand at top of image selects top floor.
        floor = min(self.floor_count - 1, max(0, int((1.0 - index_tip[1]) * self.floor_count)))
        if floor == self._floor_candidate:
            self._candidate_frames += 1
        else:
            self._floor_candidate = floor
            self._candidate_frames = 1
        if self._candidate_frames >= 4:
            events["floor"] = floor
        return events
