import logging
import math
import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from src.config import RESIZE_PERCENT, MAX_HOLDING_TIME, base_dir


def set_logger():
    return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("../logs/app.log", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def visualize_angles(image, angle, joint, width, height):
    width = int(width * RESIZE_PERCENT / 100)
    height = int(height * RESIZE_PERCENT / 100)

    if angle is not None:
        offset_x = 10
        offset_y = -10
        text_position = np.multiply(joint, [width, height]).astype(int) + np.array([offset_x, offset_y])

        cv2.putText(image, str(angle),
                    tuple(text_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 3, cv2.LINE_AA
                    )


def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None

    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_angle_vertical(point1, point2):
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    angle = np.arctan2(delta_x, delta_y) * 180.0 / np.pi
    return abs(angle)


def calculate_angle_3d(point1, point2, point3):
    if point1 is None or point2 is None or point3 is None:
        return None

    vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]])
    vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1], point3[2] - point2[2]])
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(cosine_angle) * 180.0 / np.pi
    return angle


def calculate_angle_vertical_3d(point1, point2):
    if point1 is None or point2 is None:
        return None

    vector = np.array([point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]])
    vertical_vector = np.array([0, -1, 0])
    cosine_angle = np.dot(vector, vertical_vector) / np.linalg.norm(vector)
    angle = np.arccos(cosine_angle) * 180.0 / np.pi
    return angle


def findDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def show_angles_in_box(image, l_shldr, r_shldr, trunk_incl):
    # Show angles in box
    if l_shldr is None:
        l_shldr = 'None'
    else:
        l_shldr = str("{:.2f}".format(l_shldr))

    if r_shldr is None:
        r_shldr = 'None'
    else:
        r_shldr = str("{:.2f}".format(r_shldr))

    if trunk_incl is None:
        trunk_incl = 'None'
    else:
        trunk_incl = str("{:.2f}".format(trunk_incl))

    cv2.putText(image, "Left shoulder angle: " + l_shldr,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Right shoulder angle: " + r_shldr,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Trunk angle: " + trunk_incl,
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def calculate_posture_status(values, threshold, range):
    status = []
    current_check_duration = 0
    for v in values:
        if v > threshold:
            status.append(2)
        elif range[0] <= v <= range[-1]:
            if current_check_duration < MAX_HOLDING_TIME:
                status.append(1)
                current_check_duration += 1
            else:
                status.append(2)
        else:
            status.append(0)
            current_check_duration = 0
    return status


def get_image_zoomed(ax):
    fig_new = plt.figure(figsize=(8, 4))
    ax_new = fig_new.add_axes([0.1, 0.1, 0.8, 0.8])

    for line in ax.get_lines():
        ax_new.plot(line.get_xdata(), line.get_ydata(), color=line.get_color())

    ax_new.set_title(ax.get_title(), fontsize=13)
    ax_new.set_xlim(ax.get_xlim())
    ax_new.set_ylim(ax.get_ylim())

    # trunk
    zoom_outside(ax_new, xlim=(20, 40), ylim=(0, 25), new_fig=fig_new, pos=[1., 0.75, 0.3, 0.3], color="blue")
    zoom_outside(ax_new, xlim=(180, 230), ylim=(10, 110), new_fig=fig_new, pos=[1., 0.35, 0.3, 0.3], color="red")
    zoom_outside(ax_new, xlim=(380, 420), ylim=(25, 50), new_fig=fig_new, pos=[1., -0.05, 0.3, 0.3], color="#D5D000")

    # shoulder
    '''zoom_outside(ax_new, xlim=(20, 40), ylim=(0, 25), new_fig=fig_new, pos=[1., 0.75, 0.3, 0.3], color="blue")
    zoom_outside(ax_new, xlim=(285, 320), ylim=(5, 70), new_fig=fig_new, pos=[1., 0.35, 0.3, 0.3], color="#D5D000")
    zoom_outside(ax_new, xlim=(412, 448), ylim=(100, 140), new_fig=fig_new, pos=[1., -0.05, 0.3, 0.3], color="red")'''

    fig_new.savefig(os.path.join(base_dir, 'img', 'zoomed_plot_trunk.png'), bbox_inches='tight')
    # fig_new.savefig(os.path.join(base_dir, 'img', 'zoomed_plot_shoulder.png'), bbox_inches='tight')


def zoom_outside(ax, xlim, ylim, new_fig, pos, color="red", linewidth=2, roiKwargs={}):
    roiKwargs = dict([("fill", False), ("linestyle", "dashed"),
                      ("color", color), ("linewidth", linewidth)]
                     + list(roiKwargs.items()))

    roi = [xlim[0], ylim[0], xlim[1], ylim[1]]
    ax_new = new_fig.add_axes(pos)
    ax_new.set_xlim(xlim)
    ax_new.set_ylim(ylim)

    rect = Rectangle([roi[0], roi[1]], roi[2] - roi[0], roi[3] - roi[1], **roiKwargs)
    ax.add_patch(rect)

    for line in ax.get_lines():
        ax_new.plot(line.get_xdata(), line.get_ydata(), color=line.get_color())

    ax_new.set_facecolor('#ffffff')
