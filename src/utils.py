import logging

import cv2
import numpy as np


def set_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("../logs/app.log", mode='w'),
            # logging.StreamHandler(sys.stdout)
        ]
    )


def visualize_angles(image, angle, joint):
    if angle == -1:
        pass
    else:
        cv2.putText(image, str(angle),
                    tuple(np.multiply(joint, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )


def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return -1

    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_left_arm_landmarks(mp_pose, landmarks, confidence):
    if check_visibility(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility,
                        confidence):

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        return left_shoulder, left_elbow, left_wrist

    else:
        return None, None, None


def get_right_arm_landmarks(mp_pose, landmarks, confidence):
    if check_visibility(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility,
                        confidence):

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        return right_shoulder, right_elbow, right_wrist
    else:
        return None, None, None


def get_right_leg_landmarks(mp_pose, landmarks, confidence):
    if check_visibility(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility,
                        confidence):
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        return right_hip, right_knee, right_ankle
    else:
        return None, None, None


def get_left_leg_landmarks(mp_pose, landmarks, confidence):
    if check_visibility(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility,
                        confidence):
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        return left_hip, left_knee, left_ankle
    else:
        return None, None, None


def check_visibility(a, b, c, confidence):
    return all(x > confidence for x in (a, b, c))


def show_angles_in_box(image, left_arm_angle, right_arm_angle, left_leg_angle, right_leg_angle):

    # Show angles in box
    if left_arm_angle == -1:
        left_arm_angle = 'None'
    else:
        left_arm_angle = str("{:.2f}".format(left_arm_angle))

    if right_arm_angle == -1:
        right_arm_angle = 'None'
    else:
        right_arm_angle = str("{:.2f}".format(right_arm_angle))

    if left_leg_angle == -1:
        left_leg_angle = 'None'
    else:
        left_leg_angle = str("{:.2f}".format(left_leg_angle))

    if right_leg_angle == -1:
        right_leg_angle = 'None'
    else:
        right_leg_angle = str("{:.2f}".format(right_leg_angle))

    cv2.putText(image, "Left arm angle : " + left_arm_angle,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Right arm angle : " + right_arm_angle,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Left angle knee : " + left_leg_angle,
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Right angle knee : " + right_leg_angle,
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
