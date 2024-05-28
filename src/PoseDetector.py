from datetime import datetime
import json
import logging
import os
import time

import cv2
import mediapipe as mp

from src.VideoCamera import VideoCamera
from src.config import excluded_landmarks, landmark_to_coords, SHOULDERS_ANGLE_THRESHOLD, \
    TRUNK_INCL_THRESHOLD, SHOULDERS_ANGLE_RANGE, TRUNK_INCL_RANGE, data_path, DEVICE_ID
from src.utils import visualize_angles, set_logger, show_angles_in_box, calculate_angle_vertical, calculate_angle

set_logger()


class PoseDetector:
    """
    This class is used to detect and track human poses in real-time using the MediaPipe library.

    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 enable_segmentation=False, model_complexity=1, sampling_rate=1, enable_face_landmarks=False):
        """
        Initializes the PoseDetector instance.

        :arg min_detection_confidence: The minimum detection confidence. Default is 0.5.
        :arg min_tracking_confidence: The minimum tracking confidence. Default is 0.5.
        :arg enable_segmentation: Whether to enable segmentation. Default is False.
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

        self.good_spec = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                     circle_radius=2)
        self.bad_spec = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                    circle_radius=2)
        self.check_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2,
                                                      circle_radius=2)

        self.custom_connections_style = {}
        for connection in self.custom_connections:
            self.custom_connections_style[connection] = self.good_spec

        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=self.min_detection_confidence,
                                                  min_tracking_confidence=self.min_tracking_confidence,
                                                  enable_segmentation=True,
                                                  model_complexity=self.model_complexity
                                                  )
        self.data = []
        self.data_path = data_path
        self.sampling_rate = sampling_rate

        self.cap = VideoCamera(DEVICE_ID)
        self.start_time = time.time()
        self.last_time_updated = self.start_time

        self.record_session = False

        logging.info('PoseDetector instance created.')

    def find_landmarks(self):
        """
        Turns on the camera and finds the landmarks on the human body using the MediaPipe holistic model.
        It also calculates the angles of the arms and legs and visualizes them on the image. The data is then added
        to a list.

        :return: None
        """

        if self.cap.isOpen():
            image, results = self.cap.get_frame(self.holistic)

            if results.pose_landmarks is None:
                logging.info("Ignoring empty camera frame.")
                _, png = cv2.imencode('.png', image)
                return png.tobytes()

            try:

                # Extract landmarks
                landmarks = results.pose_landmarks.landmark

                # Get segmentation mask
                if self.enable_segmentation:
                    seg_mask = cv2.cvtColor(results.segmentation_mask, cv2.COLOR_RGB2BGR)
                    image = seg_mask

                # Get coordinates of landmarks in frame defining the correspondent class attribute
                # If the landmarks is not detected, set class attribute to None
                for landmark in self.mp_holistic.PoseLandmark:
                    if landmark in landmark_to_coords:
                        if landmarks[landmark.value].visibility > self.min_detection_confidence:
                            setattr(self, landmark_to_coords[landmark], [landmarks[landmark.value].x,
                                                                         landmarks[landmark.value].y])
                        else:
                            setattr(self, landmark_to_coords[landmark], None)

                # Get joint angles
                l_shldr, r_shldr, trunk_incl = self.get_angles()

                # Visualize angles
                visualize_angles(image, l_shldr, self.left_shoulder, self.cap.width, self.cap.height)
                visualize_angles(image, r_shldr, self.right_shoulder, self.cap.width, self.cap.height)
                visualize_angles(image, trunk_incl, self.left_hip, self.cap.width, self.cap.height)

                cv2.rectangle(image, (20, 20), (220, 90), (0, 0, 0), -1)

                show_angles_in_box(image, l_shldr, r_shldr, trunk_incl)

                self.set_custom_landmark(l_shldr, r_shldr, trunk_incl)
                self.draw_landmarks(image, results)

                current_time = time.time()
                if self.record_session:
                    if current_time - self.last_time_updated >= self.sampling_rate:
                        self.data.append({
                            'timestamp': current_time,
                            'l_shldr': l_shldr,
                            'r_shldr': r_shldr,
                            'trunk_incl': trunk_incl
                        })

                        self.last_time_updated = current_time

                if self.enable_segmentation:
                    _, png = cv2.imencode('.png', 255 * seg_mask)
                else:
                    _, png = cv2.imencode('.png', image)
                return png.tobytes()

            except Exception as e:
                logging.exception(e)
                pass

    def get_angles(self):
        """
        Calculate the angles for shoulder rotation and trunk inclination based on joint positions.
        This method also handles the case where the person is seated.

        :return: Tuple containing the angles of the left shoulder, right shoulder, and trunk inclination.
                 All angles are in degrees and rounded to two decimal places.
        """

        # Calculate shoulder angles
        l_shldr = calculate_angle(self.left_elbow, self.left_shoulder, self.left_hip)
        if l_shldr is not None:
            l_shldr = round(l_shldr, 2)

        r_shldr = calculate_angle(self.right_elbow, self.right_shoulder, self.right_hip)
        if r_shldr is not None:
            r_shldr = round(r_shldr, 2)

        # Calculate trunk inclination angle
        trunk_incl = None
        if self.left_shoulder and self.left_hip and self.left_knee:
            # Check if the person is standing by comparing the x coordinates
            if abs(self.left_shoulder[0] - self.left_hip[0]) < 0.05 and abs(
                    self.left_hip[0] - self.left_knee[0]) < 0.05:
                # The person is likely standing
                trunk_incl = calculate_angle(self.left_shoulder, self.left_hip, self.left_knee)
                if trunk_incl is not None:
                    trunk_incl = 180 - trunk_incl
            else:
                # The person is likely sitting
                # Calculate the angle between the vertical and the line from the hip to the shoulder
                trunk_incl = calculate_angle_vertical(self.left_shoulder, self.left_hip)

        if trunk_incl is not None:
            trunk_incl = round(trunk_incl, 2)

        return l_shldr, r_shldr, trunk_incl

    def set_custom_landmark(self, l_shldr, r_shldr, trunk_incl):
        """
        Set custom landmark drawing specifications based on pose landmarks and specific thresholds.

        :param l_shldr: Left shoulder angle.
        :param r_shldr: Right shoulder angle.
        :param trunk_incl: Trunk inclination.
        :return: None
        """

        left_shoulder_conn = (11, 13)
        right_shoulder_conn = (12, 14)
        trunk_conn = [(12, 24), (11, 23), (23, 24)]

        for landmark in self.mp_holistic.PoseLandmark:
            if landmark in excluded_landmarks:
                self.custom_style[landmark] = self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=None)
                self.custom_connections = [connection_tuple for connection_tuple in self.custom_connections
                                           if landmark.value not in connection_tuple]

        if l_shldr is not None and l_shldr > SHOULDERS_ANGLE_THRESHOLD:
            self.custom_connections_style[left_shoulder_conn] = self.bad_spec
        elif l_shldr is not None and SHOULDERS_ANGLE_RANGE[0] <= l_shldr <= SHOULDERS_ANGLE_RANGE[-1]:
            self.custom_connections_style[left_shoulder_conn] = self.check_spec
        else:
            self.custom_connections_style[left_shoulder_conn] = self.good_spec

        if r_shldr is not None and r_shldr > SHOULDERS_ANGLE_THRESHOLD:
            self.custom_connections_style[right_shoulder_conn] = self.bad_spec
        elif r_shldr is not None and SHOULDERS_ANGLE_RANGE[0] <= r_shldr <= SHOULDERS_ANGLE_RANGE[-1]:
            self.custom_connections_style[right_shoulder_conn] = self.check_spec
        else:
            self.custom_connections_style[right_shoulder_conn] = self.good_spec

        if trunk_incl is not None and trunk_incl > TRUNK_INCL_THRESHOLD:
            for connection in trunk_conn:
                self.custom_connections_style[connection] = self.bad_spec
        elif trunk_incl is not None and TRUNK_INCL_RANGE[0] <= trunk_incl <= TRUNK_INCL_RANGE[-1]:
            for connection in trunk_conn:
                self.custom_connections_style[connection] = self.check_spec
        else:
            for connection in trunk_conn:
                self.custom_connections_style[connection] = self.good_spec

    def draw_landmarks(self, image, results):
        """
        Draw landmarks and connections on the given image based on the provided results.

        :param image: The image on which landmarks will be drawn.
        :param results: The results object containing pose and optionally face landmarks.
        :return: None
        """

        if self.enable_face_landmarks:
            self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                                           # landmark_drawing_spec=None,
                                           connection_drawing_spec=mp.solutions.drawing_styles
                                           .get_default_face_mesh_contours_style())

        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            connections=self.custom_connections,
            landmark_drawing_spec=self.custom_style,
            connection_drawing_spec=self.custom_connections_style
        )

    def save_data(self, path_to_save=data_path):
        """
        Write data collected from the pose detection to a JSON file.

        :param path_to_save: The path where the data should be saved.
        :return: None
        """

        current_time = datetime.now().strftime('%d-%m-%H-%M-%S')
        filename = f'{current_time}.json'

        with open(os.path.join(data_path, filename), 'w') as f:
            logging.info('Save data in %s', path_to_save)
            json.dump(self.data, f)
