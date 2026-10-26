import os
import sys
import unittest
import time
import numpy as np
import cv2

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.alert_system import AlertSystem
from modules.eye_detection import calculate_ear, detect_eyes, predict_eye_status, load_eye_cascade
from modules.yawning_detection import calculate_mar, load_face_cascade
from modules.drowsiness_detector import ComprehensiveDrowsinessDetector


class TestDrowsinessComponents(unittest.TestCase):

    def test_alert_system_resolution(self):
        alert = AlertSystem(eye_closed_sec=1.0)
        self.assertIsNotNone(alert.resolved_path)
        self.assertTrue(os.path.isfile(alert.resolved_path))
        print("AlertSystem sound resolved to:", alert.resolved_path)

        # Test state transitions
        res1 = alert.update("Eyes Open")
        self.assertEqual(res1["alert_level"], "NORMAL")
        self.assertFalse(res1["alarm_playing"])

    def test_alert_system_timing(self):
        # Verify that eyes closed triggers alert accurately after threshold
        alert = AlertSystem(eye_closed_sec=0.5, tolerance_sec=0.2)
        alert.sound_enabled = False  # silence in unit test

        # Start closed
        res = alert.update("Eyes Closed")
        self.assertEqual(res["alert_level"], "NORMAL")

        # Wait 0.6 seconds
        time.sleep(0.6)
        res_after = alert.update("Eyes Closed")
        self.assertEqual(res_after["alert_level"], "CRITICAL")
        self.assertTrue(res_after["alarm_playing"])
        self.assertGreaterEqual(res_after["closed_duration"], 0.5)
        print("Timing verified: Triggered CRITICAL alarm after", res_after["closed_duration"], "seconds")

        # Reopening eyes stops alarm
        alert.last_closed_time = time.time() - 0.5  # beyond tolerance
        res_open = alert.update("Eyes Open")
        self.assertEqual(res_open["alert_level"], "NORMAL")
        self.assertFalse(res_open["alarm_playing"])

    def test_ear_calculation(self):
        # Open eye coordinates
        open_eye = [
            (10, 30),  # p1: outer corner
            (20, 20),  # p2: top 1
            (30, 20),  # p3: top 2
            (40, 30),  # p4: inner corner
            (30, 40),  # p5: bottom 2
            (20, 40)   # p6: bottom 1
        ]
        ear_open = calculate_ear(open_eye)
        self.assertGreater(ear_open, 0.25)

        # Closed eye coordinates (collapsed vertical distance)
        closed_eye = [
            (10, 30),
            (20, 29),
            (30, 29),
            (40, 30),
            (30, 31),
            (20, 31)
        ]
        ear_closed = calculate_ear(closed_eye)
        self.assertLess(ear_closed, 0.15)
        print("Calculated Open Eye EAR:", ear_open, "Closed EAR:", ear_closed)

    def test_mar_calculation(self):
        # Normal mouth: vertical 10, horizontal 40 -> MAR = 0.25
        mar_normal = calculate_mar((20, 25), (20, 35), (0, 30), (40, 30))
        self.assertAlmostEqual(mar_normal, 0.25, places=2)

        # Yawning mouth: vertical 35, horizontal 40 -> MAR = 0.875
        mar_yawn = calculate_mar((20, 10), (20, 45), (0, 30), (40, 30))
        self.assertGreater(mar_yawn, 0.70)
        print("Calculated Normal MAR:", mar_normal, "Yawn MAR:", mar_yawn)

    def test_detector_processing(self):
        detector = ComprehensiveDrowsinessDetector(
            ear_threshold=0.22,
            mar_threshold=0.60,
            eye_closed_sec=1.0,
            model_type="facemesh",
            enable_sound=False
        )

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, metrics = detector.process_frame(dummy)

        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertIn("face_detected", metrics)
        self.assertIn("eye_status", metrics)
        self.assertIn("drowsiness_score", metrics)
        detector.close()


if __name__ == "__main__":
    unittest.main()
