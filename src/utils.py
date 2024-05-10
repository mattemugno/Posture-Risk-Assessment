import logging
import math
import sys

import cv2
import numpy as np


def set_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("../logs/app.log", mode='w'),
            logging.StreamHandler(sys.stdout)
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


def findDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def show_angles_in_box(image, l_shldr, r_shldr, trunk_incl):
    # Show angles in box
    if l_shldr == -1:
        l_shldr = 'None'
    else:
        l_shldr = str("{:.2f}".format(l_shldr))

    if r_shldr == -1:
        r_shldr = 'None'
    else:
        r_shldr = str("{:.2f}".format(r_shldr))

    if trunk_incl == -1:
        trunk_incl = 'None'
    else:
        trunk_incl = str("{:.2f}".format(trunk_incl))

    cv2.putText(image, "Left shoulder angle: " + l_shldr,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Right shoulder angle: " + r_shldr,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Trunk inclination: " + trunk_incl,
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)