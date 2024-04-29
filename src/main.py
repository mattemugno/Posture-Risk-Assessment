import json
import logging
import os
import time

import cv2
import mediapipe as mp

from src.utils import set_logger, calculate_angle, get_left_arm_landmarks, get_right_arm_landmarks, \
    get_right_leg_landmarks, visualize_angles, get_left_leg_landmarks

set_logger()
data = []
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
data_json = os.path.join(base_dir, 'data', 'data.json')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

logging.info("Turning on the camera...")
cap = cv2.VideoCapture(0)

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            seg_mask = results.segmentation_mask

            # Get coordinates
            left_shoulder, left_elbow, left_wrist = get_left_arm_landmarks(mp_pose, landmarks)
            right_shoulder, right_elbow, right_wrist = get_right_arm_landmarks(mp_pose, landmarks)
            right_hip, right_knee, right_ankle = get_right_leg_landmarks(mp_pose, landmarks)
            left_hip, left_knee, left_ankle = get_left_leg_landmarks(mp_pose, landmarks)

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

        except Exception as e:
            logging.exception(e)
            pass

        cv2.rectangle(image, (20, 20), (220, 110), (0, 0, 0), -1)

        # Show angles in box
        cv2.putText(image, "Left arm angle : " + str("{:.2f}".format(left_arm_angle)),
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, "Right arm angle : " + str("{:.2f}".format(right_arm_angle)),
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, "Left angle knee : " + str("{:.2f}".format(left_leg_angle)),
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, "Right angle knee : " + str("{:.2f}".format(right_leg_angle)),
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Landmarks feed', image)

        cv2.imshow('Segmentation mask', seg_mask)

        # Add the data to the list
        data.append({
            'timestamp': time.time(),
            'left_arm_angle': left_arm_angle,
            'right_arm_angle': right_arm_angle,
            'right_leg_angle': right_leg_angle,
            'left_leg_angle': left_leg_angle
        })

        if cv2.waitKey(10) & 0xFF == ord('q'):
            logging.info("Exit the camera")
            break

    cap.release()
    cv2.destroyAllWindows()

with open(data_json, 'w') as f:
    logging.info('Save data in %s', data_json)
    json.dump(data, f)
