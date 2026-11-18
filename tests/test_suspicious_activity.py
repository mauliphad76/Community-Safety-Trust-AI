import unittest

import numpy as np

from modules.suspicious_activity import (
    claim_alert_event,
    detect_frame,
    should_play_alarm,
)


class MockBox:
    def __init__(self, class_id, confidence, coordinates):
        self.cls = np.array([class_id])
        self.conf = np.array([confidence])
        self.xyxy = np.array([coordinates])


class MockDetector:
    names = {0: "person", 1: "bicycle", 43: "knife"}

    def __init__(self, detections):
        self.detections = detections
        self.predict_arguments = None

    def predict(self, **kwargs):
        self.predict_arguments = kwargs
        return [type("MockResult", (), {"boxes": self.detections})()]


class SuspiciousActivityTests(unittest.TestCase):
    def test_processes_every_detection_returned_by_yolo(self):
        detector = MockDetector([
            MockBox(1, 0.30 + index / 100, [index, 1, index + 1, 10])
            for index in range(12)
        ])

        _, _, detections, _ = detect_frame(
            np.full((48, 48, 3), 120, dtype=np.uint8),
            detector,
            confidence_threshold=0.18,
        )

        self.assertEqual(len(detections), 12)
        self.assertEqual(
            [item["box"][0] for item in detections],
            list(range(12)),
        )

    def test_knife_alert_uses_loaded_class_and_independent_threshold(self):
        detector = MockDetector([
            MockBox(0, 0.80, [1, 1, 10, 10]),
            MockBox(1, 0.40, [12, 1, 20, 10]),
            MockBox(43, 0.72, [22, 1, 30, 10]),
            MockBox(43, 0.60, [32, 1, 40, 10]),
            MockBox(1, 0.10, [2, 32, 8, 40]),
        ])

        annotated, people, visible, knives = detect_frame(
            np.full((48, 48, 3), 120, dtype=np.uint8),
            detector,
            confidence_threshold=0.25,
            knife_class_id=43,
            knife_confidence_threshold=0.65,
            image_size=960,
            iou_threshold=0.75,
            max_detections=300,
        )

        self.assertIsNone(detector.predict_arguments["classes"])
        self.assertEqual(detector.predict_arguments["conf"], 0.25)
        self.assertEqual(detector.predict_arguments["imgsz"], 960)
        self.assertEqual(detector.predict_arguments["iou"], 0.75)
        self.assertEqual(detector.predict_arguments["max_det"], 300)
        self.assertEqual(people, 1)
        self.assertEqual(
            [item["class_name"] for item in visible],
            ["person", "bicycle", "knife", "knife"],
        )
        self.assertEqual(len(knives), 1)
        self.assertAlmostEqual(knives[0]["confidence"], 0.72)
        self.assertEqual(annotated.shape, (48, 48, 3))

    def test_alarm_cooldown_gates_sounds_but_not_alert_detections(self):
        self.assertTrue(should_play_alarm(True, True, 20.0, None, 10))
        self.assertFalse(should_play_alarm(True, True, 25.0, 20.0, 10))
        self.assertTrue(should_play_alarm(True, True, 30.0, 20.0, 10))
        self.assertFalse(should_play_alarm(False, True, 30.0, None, 10))
        self.assertFalse(should_play_alarm(True, False, 30.0, None, 10))

    def test_duplicate_alert_event_is_claimed_once_and_history_is_bounded(self):
        recent_events = []
        uploaded_events = set()

        self.assertTrue(claim_alert_event("image:abc", recent_events))
        self.assertFalse(claim_alert_event("image:abc", recent_events))
        for index in range(130):
            claim_alert_event(f"camera:{index}", recent_events, max_events=128)

        self.assertEqual(len(recent_events), 128)
        self.assertNotIn("image:abc", recent_events)
        self.assertTrue(claim_alert_event("image:abc", recent_events))
        self.assertTrue(claim_alert_event("image:abc", uploaded_events))
        self.assertFalse(claim_alert_event("image:abc", uploaded_events))


if __name__ == "__main__":
    unittest.main()
