import json
import logging
import os
import time

import cv2
import mediapipe as mp

from src.utils import get_left_arm_landmarks, get_right_arm_landmarks, get_right_leg_landmarks, get_left_leg_landmarks, \
    calculate_angle, visualize_angles, set_logger, show_angles_in_box

set_logger()
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
data_path = os.path.join(base_dir, 'data', 'data.json')


class PoseDetector:
    """
    This class is used to detect and track human poses in real-time using the MediaPipe library.

    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 enable_segmentation=True, model_complexity=1):
        """
        Initializes the PoseDetector instance.

        Args:
            min_detection_confidence (float): The minimum detection confidence. Default is 0.5.
            min_tracking_confidence (float): The minimum tracking confidence. Default is 0.5.
            enable_segmentation (bool): Whether to enable segmentation. Default is True.
            model_complexity (int): The complexity of the model to be used. Default is 1.
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

        logging.info('PoseDetector instance created.')

    def find_landmarks(self, device_id=0):
        """
            This method turns on the camera and finds the landmarks on the human body using the MediaPipe holistic model.
            It also calculates the angles of the arms and legs and visualizes them on the image. The data is then added
            to a list.

            Parameters:
                device_id (int): The ID of the device from which the video is to be captured. Default is 0,
                which usually corresponds to the built-in webcam

            Returns:
                None
        """

        logging.info("Turning on the camera for finding landmarks")
        cap = cv2.VideoCapture(device_id)
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

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                seg_mask = cv2.cvtColor(results.segmentation_mask, cv2.COLOR_BGR2RGB)

                # Get coordinates
                left_shoulder, left_elbow, left_wrist = get_left_arm_landmarks(self.mp_holistic, landmarks)
                right_shoulder, right_elbow, right_wrist = get_right_arm_landmarks(self.mp_holistic, landmarks)
                right_hip, right_knee, right_ankle = get_right_leg_landmarks(self.mp_holistic, landmarks)
                left_hip, left_knee, left_ankle = get_left_leg_landmarks(self.mp_holistic, landmarks)

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

                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                           circle_radius=2),
                                               self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                           circle_radius=2)
                                               )

                cv2.imshow('Landmarks feed', image)
                cv2.imshow('Segmentation mask', seg_mask)

                # Add the data to the list
                self.data.append({
                    'timestamp': time.time(),
                    'left_arm_angle': left_arm_angle,
                    'right_arm_angle': right_arm_angle,
                    'right_leg_angle': right_leg_angle,
                    'left_leg_angle': left_leg_angle
                })

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
            This method writes the data collected from the pose detection to a JSON file.

            Parameters:
                path_to_save (str): The path where the data should be saved.

            Returns:
                None.
        """
        with open(path_to_save, 'w') as f:
            logging.info('Save data in %s', path_to_save)
            json.dump(self.data, f)
