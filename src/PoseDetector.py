import json
import logging
import os
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from src.VideoCamera import VideoCamera
from src.config import excluded_landmarks, landmark_to_coords, ARM_ANGLE_THRESHOLD
from src.utils import calculate_angle, visualize_angles, set_logger, show_angles_in_box, recolor_image

matplotlib.use('TkAgg')

set_logger()
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
data_path = os.path.join(base_dir, 'data', 'data.json')


class PoseDetector:
    """
    This class is used to detect and track human poses in real-time using the MediaPipe library.

    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 enable_segmentation=True, model_complexity=1, sampling_rate=1.0, enable_face_landmarks=False):
        """
        Initializes the PoseDetector instance.

        :arg min_detection_confidence: The minimum detection confidence. Default is 0.5.
        :arg min_tracking_confidence: The minimum tracking confidence. Default is 0.5.
        :arg enable_segmentation: Whether to enable segmentation. Default is True.
        :arg model_complexity: The complexity of the model to be used. Default is 1.
        :arg sampling_rate: The frequency with which angles are detected. Default is 1 second.
        :arg enable_face_landmarks: Whether to show or not face landmarks. Default is False.
        """

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_segmentation = enable_segmentation
        self.enable_face_landmarks = enable_face_landmarks
        self.model_complexity = model_complexity

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.custom_style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        self.custom_connections = list(self.mp_holistic.POSE_CONNECTIONS)

        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=self.min_detection_confidence,
                                                  min_tracking_confidence=self.min_tracking_confidence,
                                                  enable_segmentation=self.enable_segmentation,
                                                  model_complexity=self.model_complexity
                                                  )
        self.data = []
        self.data_path = data_path
        self.sampling_rate = sampling_rate

        logging.info('PoseDetector instance created.')

    def find_landmarks(self, device_id=0):
        """
        Turns on the camera and finds the landmarks on the human body using the MediaPipe holistic model.
        It also calculates the angles of the arms and legs and visualizes them on the image. The data is then added
        to a list.

        :param device_id: The ID of the device from which the video is to be captured. Default is 0,
                which usually corresponds to the built-in webcam
        :return: None
        """

        logging.info("Turning on the camera for finding landmarks")
        cap = VideoCamera(0)
        start_time = time.time()
        last_update_time = start_time

        for landmark in self.mp_holistic.PoseLandmark:
            if landmark in excluded_landmarks:
                self.custom_style[landmark] = self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=None)
                self.custom_connections = [connection_tuple for connection_tuple in self.custom_connections
                                           if landmark.value not in connection_tuple]
            else:
                self.custom_style[landmark] = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                          circle_radius=2)

        while cap.isOpen():
            success, image = cap.get_frame()

            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Make detection
            results = self.holistic.process(image)

            # Recolor image
            image = recolor_image(image)

            try:

                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                seg_mask = cv2.cvtColor(results.segmentation_mask, cv2.COLOR_BGR2RGB)

                # Get coordinates of landmarks in frame defining the correspondent class attribute
                # If the landmarks is not detected, set class attribute to None
                for landmark in self.mp_holistic.PoseLandmark:
                    if landmark in landmark_to_coords:
                        if landmarks[landmark.value].visibility > self.min_detection_confidence:
                            setattr(self, landmark_to_coords[landmark], [landmarks[landmark.value].x,
                                                                         landmarks[landmark.value].y])
                        else:
                            setattr(self, landmark_to_coords[landmark], None)

                # Get shoulders' angles
                l_shldr = calculate_angle(self.left_elbow, self.left_shoulder, self.left_hip)
                r_shldr = calculate_angle(self.right_elbow, self.right_shoulder, self.right_hip)

                # Get trunk inclination
                trunk_incl = calculate_angle(self.left_shoulder, self.left_hip, self.left_knee)

                # neck_incl = calculate_angle()
                ### neck inclination

                # Visualize angles
                visualize_angles(image, l_shldr, self.left_shoulder)
                visualize_angles(image, r_shldr, self.right_shoulder)
                visualize_angles(image, trunk_incl, self.left_hip)

                cv2.rectangle(image, (20, 20), (220, 70), (0, 0, 0), -1)

                show_angles_in_box(image, l_shldr, r_shldr)

                # Render detections
                if self.enable_face_landmarks:
                    self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                                                   landmark_drawing_spec=None,
                                                   connection_drawing_spec=mp.solutions.drawing_styles
                                                   .get_default_face_mesh_contours_style())

                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    connections=self.custom_connections,
                    landmark_drawing_spec=self.custom_style,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                        circle_radius=2)
                )

                cv2.imshow('Landmarks feed', image)

                if self.enable_segmentation:
                    cv2.imshow('Segmentation mask', seg_mask)

                current_time = time.time()
                if current_time - last_update_time >= self.sampling_rate:
                    self.data.append({
                        'timestamp': current_time - start_time,
                        'l_shldr': l_shldr,
                        'r_shldr': r_shldr,
                        'trunk_incl': trunk_incl
                    })

                    last_update_time = current_time

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    logging.info("Exit the camera")
                    break

            except Exception as e:
                logging.exception(e)
                pass

    def save_data(self, path_to_save=data_path):
        """
        Write data collected from the pose detection to a JSON file.

        :param path_to_save: The path where the data should be saved.
        :return: None
        """
        with open(path_to_save, 'w') as f:
            logging.info('Save data in %s', path_to_save)
            json.dump(self.data, f)

    def plot_posture(self):
        """
        Plot posture angles from data.json

        :return: None
        """

        data = json.load(open(self.data_path, 'r'))
        x = [v['timestamp'] for v in data]
        y_right_arm = [v['r_shldr'] for v in data]
        y_left_arm = [v['l_shldr'] for v in data]
        # trunk = [v['trunk_incl'] for v in data]

        plt.style.use('seaborn-v0_8')
        plt.grid(True)

        plt.plot(x, y_right_arm, color='blue', marker='_', label='Right arm angle', mfc='red', mec='k')
        plt.plot(x, y_left_arm, color='green', marker='_', label='Left arm angle', mfc='red', mec='k')

        plt.xlim(left=0)
        plt.ylim(top=180)
        plt.axhline(y=ARM_ANGLE_THRESHOLD, color='r', linestyle='--', label='Arm threshold')

        plt.fill_between(x, y_right_arm, np.maximum(y_right_arm, 45),
                         where=(np.array(y_right_arm) > 45), color='red', alpha=0.3)

        plt.xlabel('Time', fontsize='small')
        plt.ylabel('Angles', fontsize='small')

        plt.title('Angles in time')
        plt.legend(fontsize='small')
        plt.tick_params(axis='both', labelsize=7)

        plt.legend()
        plt.show()
