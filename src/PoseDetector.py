import io
import json
import logging
import os
import time

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

from src.VideoCamera import VideoCamera
from src.config import excluded_landmarks, landmark_to_coords
from src.utils import calculate_angle, visualize_angles, set_logger, show_angles_in_box

set_logger()
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
data_path = os.path.join(base_dir, 'data')
model_path = os.path.join(base_dir, 'src', 'pose_landmarker.task')


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
                                                  enable_segmentation=True,
                                                  model_complexity=self.model_complexity
                                                  )
        self.data = []
        self.data_path = data_path
        self.sampling_rate = sampling_rate

        self.cap = VideoCamera(0)
        self.start_time = time.time()
        self.last_time_updated = self.start_time

        logging.info('PoseDetector instance created.')

    def find_landmarks(self):
        """
        Turns on the camera and finds the landmarks on the human body using the MediaPipe holistic model.
        It also calculates the angles of the arms and legs and visualizes them on the image. The data is then added
        to a list.

        :return: None
        """

        self.set_custom_landmark()

        if self.cap.isOpen():
            image, results = self.cap.get_frame(self.holistic)

            if results is None:
                logging.info("Ignoring empty camera frame.")
                return None

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
                visualize_angles(image, l_shldr, self.left_shoulder)
                visualize_angles(image, r_shldr, self.right_shoulder)
                visualize_angles(image, trunk_incl, self.left_hip)

                cv2.rectangle(image, (20, 20), (220, 90), (0, 0, 0), -1)

                show_angles_in_box(image, l_shldr, r_shldr, trunk_incl)

                self.draw_landmarks(image, results)

                current_time = time.time()
                if current_time - self.last_time_updated >= self.sampling_rate:
                    self.data.append({
                        'timestamp': current_time - self.start_time,
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
        # Get shoulders' angles
        l_shldr = calculate_angle(self.left_elbow, self.left_shoulder, self.left_hip)
        r_shldr = calculate_angle(self.right_elbow, self.right_shoulder, self.right_hip)
        # Get trunk inclination
        trunk_incl = calculate_angle(self.left_shoulder, self.left_hip, self.left_knee)
        return l_shldr, r_shldr, trunk_incl

    def set_custom_landmark(self):
        for landmark in self.mp_holistic.PoseLandmark:
            if landmark in excluded_landmarks:
                self.custom_style[landmark] = self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=None)
                self.custom_connections = [connection_tuple for connection_tuple in self.custom_connections
                                           if landmark.value not in connection_tuple]
            else:
                self.custom_style[landmark] = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                          circle_radius=2)

    def draw_landmarks(self, image, results):
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

    def save_data(self, path_to_save=data_path):
        """
        Write data collected from the pose detection to a JSON file.

        :param path_to_save: The path where the data should be saved.
        :return: None
        """
        with open(os.path.join(data_path, 'data.json'), 'w') as f:
            logging.info('Save data in %s', path_to_save)
            json.dump(self.data, f)

    def plot_posture(self):
        """
        Plot posture angles from data.json

        :return: BytesIO object containing the plot image
        """
        with open(os.path.join(self.data_path, 'data.json'), 'r') as file:
            data = json.load(file)
            x = [v['timestamp'] for v in data]
            y_right_arm = [v['r_shldr'] for v in data]
            y_left_arm = [v['l_shldr'] for v in data]
            trunk = [v['trunk_incl'] for v in data]

            plt.switch_backend('agg')

            fig, axs = plt.subplots(3, 1, sharex=True)

            axs[0].plot(x, y_right_arm, color='blue', label='Right shoulder angle', mfc='red')
            axs[0].set_ylabel('Angles')
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(x, y_left_arm, color='green', label='Left shoulder angle')
            axs[1].set_ylabel('Angles')
            axs[1].legend()
            axs[1].grid(True)

            axs[2].plot(x, trunk, color='red', label='Trunk angle')
            axs[2].set_xlabel('Time')
            axs[2].set_ylabel('Angles')
            axs[2].legend()
            axs[2].grid(True)

            plt.xlim(left=0)
            plt.ylim(top=180)

            plt.suptitle('Angles over Time', y=0.92)

            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            plt.close()

            return img_buffer


