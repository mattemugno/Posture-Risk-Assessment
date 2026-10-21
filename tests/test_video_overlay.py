import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

from src.PoseDetector import PoseDetector
from src.utils import show_angles_in_box, visualize_angles


class VideoOverlayTests(unittest.TestCase):
    def test_labels_preserve_frame_format_and_fit_at_edges(self):
        for shape in ((240, 320, 3), (720, 1280, 3)):
            for dtype, background in ((np.uint8, 64), (np.float32, 0.25)):
                with self.subTest(shape=shape, dtype=dtype):
                    image = np.full(shape, background, dtype=dtype)
                    show_angles_in_box(image, 12.34, 65.43, 10)
                    visualize_angles(image, 65.43, (1, 1), shape[1], shape[0])
                    self.assertEqual(image.shape, shape)
                    self.assertEqual(image.dtype, dtype)
                    self.assertTrue(np.isfinite(image).all())
                    np.testing.assert_array_equal(image[shape[0] // 2, shape[1] // 2],
                                                  [background] * 3)
                    if dtype == np.float32:
                        self.assertGreaterEqual(image.min(), 0)
                        self.assertLessEqual(image.max(), 1)
                    self.assertTrue(np.any(image != background))

    def test_missing_measurements_do_not_draw_joint_labels(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        visualize_angles(image, None, None, 320, 240)
        visualize_angles(image, 10, None, 320, 240)
        self.assertFalse(image.any())
        show_angles_in_box(image, None, None, None)
        self.assertTrue(image.any())

    def test_normal_and_segmentation_frames_keep_recorded_measurements(self):
        landmarks = [SimpleNamespace(x=0.6, y=0.6, visibility=1) for _ in range(33)]
        results = SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=landmarks),
            segmentation_mask=np.full((264, 352), 0.25, dtype=np.float32),
        )
        for segmentation in (False, True):
            with self.subTest(segmentation=segmentation):
                pose = PoseDetector.__new__(PoseDetector)
                pose.cap = SimpleNamespace(
                    isOpen=lambda: True, width=320, height=240,
                    get_frame=lambda _: (np.full((264, 352, 3), 64, dtype=np.uint8), results),
                )
                pose.holistic = object()
                pose.mp_holistic = SimpleNamespace(PoseLandmark=PoseLandmark)
                pose.min_detection_confidence = 0.5
                pose.enable_segmentation = segmentation
                pose.get_angles = lambda: (12.34, 65.43, 10)
                pose.set_custom_landmark = Mock()
                pose.draw_landmarks = Mock()
                pose.record_session = True
                pose.last_time_updated = 0
                pose.sampling_rate = 1
                pose.data = []
                with patch('src.PoseDetector.time.time', return_value=100):
                    encoded = pose.find_landmarks()
                self.assertIsNotNone(encoded)
                frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(frame.shape, (264, 352, 3))
                self.assertEqual(frame.dtype, np.uint8)
                self.assertEqual(pose.data, [{
                    'timestamp': 100, 'l_shldr': 12.34, 'r_shldr': 65.43, 'trunk_incl': 10,
                }])
                pose.set_custom_landmark.assert_called_once_with(12.34, 65.43, 10)


if __name__ == '__main__':
    unittest.main()
