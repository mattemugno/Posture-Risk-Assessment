import datetime
import json
import logging
import os
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from src.utils import get_left_arm_landmarks, get_right_arm_landmarks, get_right_leg_landmarks, get_left_leg_landmarks, \
    calculate_angle, visualize_angles, set_logger, show_angles_in_box

matplotlib.use('TkAgg')

set_logger()
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
data_path = os.path.join(base_dir, 'data', 'data.json')


class PoseDetector:
    """
    This class is used to detect and track human poses in real-time using the MediaPipe library.

    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 enable_segmentation=True, model_complexity=1, sampling_rate=1.0):
        """
        Initializes the PoseDetector instance.

        :arg min_detection_confidence: The minimum detection confidence. Default is 0.5.
        :arg min_tracking_confidence: The minimum tracking confidence. Default is 0.5.
        :arg enable_segmentation: Whether to enable segmentation. Default is True.
        :arg model_complexity: The complexity of the model to be used. Default is 1.
        :arg sampling_rate: The frequency with which angles are detected. Default is 1 second.
        """

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_segmentation = enable_segmentation
        self.model_complexity = model_complexity
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
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
        cap = cv2.VideoCapture(device_id)
        start_time = time.time()
        last_update_time = start_time

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = self.holistic.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:

                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                seg_mask = cv2.cvtColor(results.segmentation_mask, cv2.COLOR_BGR2RGB)

                # Get coordinates
                left_shoulder, left_elbow, left_wrist = get_left_arm_landmarks(self.mp_holistic, landmarks,
                                                                               self.min_tracking_confidence)
                right_shoulder, right_elbow, right_wrist = get_right_arm_landmarks(self.mp_holistic, landmarks,
                                                                                   self.min_tracking_confidence)
                right_hip, right_knee, right_ankle = get_right_leg_landmarks(self.mp_holistic, landmarks,
                                                                             self.min_tracking_confidence)
                left_hip, left_knee, left_ankle = get_left_leg_landmarks(self.mp_holistic, landmarks,
                                                                         self.min_tracking_confidence)

                # Calculate angles
                left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)

                # Visualize angles
                visualize_angles(image, left_arm_angle, left_elbow)
                visualize_angles(image, right_arm_angle, right_elbow)
                visualize_angles(image, right_leg_angle, right_knee)
                visualize_angles(image, left_leg_angle, left_knee)

                cv2.rectangle(image, (20, 20), (220, 110), (0, 0, 0), -1)

                show_angles_in_box(image, left_arm_angle, right_arm_angle, left_leg_angle, right_leg_angle)

                # Render detections
                # self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                #                                landmark_drawing_spec=None,
                #                                connection_drawing_spec=mp.solutions.drawing_styles
                #                                .get_default_face_mesh_contours_style())

                print(results.pose_landmarks)
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                           circle_radius=2),
                                               self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                           circle_radius=2)
                                               )

                cv2.imshow('Landmarks feed', image)
                cv2.imshow('Segmentation mask', seg_mask)

                current_time = time.time()
                if current_time - last_update_time >= self.sampling_rate:

                    self.data.append({
                        'timestamp': current_time - start_time,
                        'left_arm_angle': left_arm_angle,
                        'right_arm_angle': right_arm_angle,
                        'right_leg_angle': right_leg_angle,
                        'left_leg_angle': left_leg_angle
                    })

                    last_update_time = current_time

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    logging.info("Exit the camera")
                    break

            except Exception as e:
                logging.exception(e)
                pass

        cap.release()
        cv2.destroyAllWindows()

    def write_data(self, path_to_save=data_path):
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
        y_right_arm = [v['right_arm_angle'] for v in data]
        y_left_arm = [v['left_arm_angle'] for v in data]
        y_right_leg = [v['right_leg_angle'] for v in data]
        y_left_leg = [v['left_leg_angle'] for v in data]

        plt.style.use('seaborn-v0_8')
        plt.grid(True)

        plt.plot(x, y_right_arm, color='magenta', marker='d', label='Right arm angle', mfc='red', mec='k')
        plt.plot(x, y_left_arm, color='green', marker='d', label='Left arm angle', mfc='red', mec='k')
        plt.plot(x, y_right_leg, color='blue', marker='d', label='Right leg angle', mfc='red', mec='k')
        plt.plot(x, y_left_leg, color='yellow', marker='d', label='Left leg angle', mfc='red', mec='k')

        plt.xlim(left=0)
        plt.ylim(top=180)
        plt.axhline(y=45, color='r', linestyle='--', label='Arm threshold')
        plt.axhline(y=30, color='g', linestyle='--', label='Leg threshold')

        plt.fill_between(x, y_right_arm, np.maximum(y_right_arm, 45),
                         where=(np.array(y_right_arm) > 45), color='red', alpha=0.3)
        plt.fill_between(x, y_right_leg, np.maximum(y_right_leg, 45),
                         where=(np.array(y_right_leg) > 45), color='green', alpha=0.3)

        plt.xlabel('Time', fontsize='small')
        plt.ylabel('Angles', fontsize='small')

        plt.title('Angles in time')
        plt.legend(fontsize='small')
        plt.tick_params(axis='both', labelsize=7)

        plt.legend()
        plt.show()
