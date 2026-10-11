"""Unit tests for mapping MediaPipe hand landmarks into application events."""

import unittest

from src.gesture_controller import GestureController


def landmarks(index_tip=(0.5, 0.5), thumb_tip=(0.2, 0.2)):
    points = [(0.0, 0.0, 0.0) for _ in range(21)]
    points[8] = (*index_tip, 0.0)
    points[4] = (*thumb_tip, 0.0)
    return points


class GestureControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = GestureController(floor_count=3)

    def test_stable_index_position_selects_expected_floor(self):
        hand = landmarks(index_tip=(0.5, 0.1))

        events = {}
        for _ in range(4):
            events = self.controller.update(hand)

        self.assertEqual(events["floor"], 2)

    def test_floor_selection_is_clamped_to_building(self):
        upper_hand = landmarks(index_tip=(0.5, 0.0))
        lower_hand = landmarks(index_tip=(0.5, 1.0))

        for _ in range(4):
            upper_events = self.controller.update(upper_hand)
        self.assertEqual(upper_events["floor"], 2)

        for _ in range(4):
            lower_events = self.controller.update(lower_hand)
        self.assertEqual(lower_events["floor"], 0)

    def test_pinch_toggles_once_until_released(self):
        pinched = landmarks(index_tip=(0.40, 0.40), thumb_tip=(0.42, 0.42))
        open_hand = landmarks(index_tip=(0.5, 0.5), thumb_tip=(0.2, 0.2))

        self.assertTrue(self.controller.update(pinched)["toggle_running"])
        self.assertFalse(self.controller.update(pinched)["toggle_running"])
        self.assertFalse(self.controller.update(open_hand)["toggle_running"])
        self.assertTrue(self.controller.update(pinched)["toggle_running"])

    def test_missing_hand_returns_no_events(self):
        self.assertEqual(self.controller.update(None), {})
        self.assertEqual(self.controller.update([]), {})


if __name__ == "__main__":
    unittest.main()
