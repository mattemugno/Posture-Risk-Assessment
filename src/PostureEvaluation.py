import json
import os

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.image import imread
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from src.config import MAX_HOLDING_TIME, SHOULDERS_ANGLE_RANGE, TRUNK_INCL_RANGE, \
    TRUNK_INCL_THRESHOLD, SHOULDERS_ANGLE_THRESHOLD
from src.utils import set_logger, calculate_posture_status

set_logger()
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


### Trunk inclination -- alfa angle
### Shoulder's angles

class PostureEvaluation:
    """
    This class is used to assess posture angles and show the risks associated with different parts of the body.
    """

    def __init__(self, data_path):
        """
        Initializes the PostureEvaluation instance
        :arg data_path: Path of the json file
        """
        self.data_path = data_path

        with open(self.data_path, 'r') as file:
            data = json.load(file)
            data = [v for v in data if
                    v['r_shldr'] is not None and v['l_shldr'] is not None and v['trunk_incl'] is not None]
            self.time = [v['timestamp'] for v in data]
            self.r_shldr = [v['r_shldr'] for v in data]
            self.l_shldr = [v['l_shldr'] for v in data]
            self.trunk = [v['trunk_incl'] for v in data]

        self.max_holding_time = MAX_HOLDING_TIME
        self.shoulders_angle_range = SHOULDERS_ANGLE_RANGE
        self.trunk_incl_range = TRUNK_INCL_RANGE

        self.not_acceptable_percentages = None

        self.not_acceptable_shldr_r = 0
        self.not_acceptable_shldr_l = 0
        self.not_acceptable_count_trunk = 0
        plt.rcParams['axes.facecolor'] = '#ffffff'

    def calculate_percentages(self):
        """
        Calculate the percentage of time spent in a bad posture for each shoulder

        :return: Bad posture percentage based on each shoulder's and trunk angle
        """

        holding_time_r = 0
        holding_time_l = 0
        holding_time_t = 0

        time_r = 0
        time_l = 0
        time_t = 0

        for gamma_r, gamma_l, alfa in zip(self.r_shldr, self.l_shldr, self.trunk):

            # 0 < gamma < 20 acceptable
            # 20 < gamma < 60 check holding time
            # gamma > 60 not acceptable
            if gamma_r is not None:
                time_r += 1
                if self.shoulders_angle_range[0] <= gamma_r <= self.shoulders_angle_range[-1]:
                    holding_time_r += 1
                    if holding_time_r >= self.max_holding_time:
                        self.not_acceptable_shldr_r += 1
                        holding_time_r = 0
                elif gamma_r > SHOULDERS_ANGLE_THRESHOLD:
                    self.not_acceptable_shldr_r += 1
                else:
                    holding_time_r = 0

            if gamma_l is not None:
                time_l += 1
                if self.shoulders_angle_range[0] <= gamma_l <= self.shoulders_angle_range[-1]:
                    holding_time_l += 1
                    if holding_time_l >= self.max_holding_time:
                        self.not_acceptable_shldr_l += 1
                        holding_time_l = 0
                elif gamma_l > SHOULDERS_ANGLE_THRESHOLD:
                    self.not_acceptable_shldr_l += 1
                else:
                    holding_time_l = 0

            # 0 < alfa < 20 acceptable
            # 20 < alfa < 60 check holding time
            # alfa > 60 not acceptable
            if alfa is not None:
                time_t += 1
                if self.trunk_incl_range[0] <= alfa <= self.trunk_incl_range[-1]:
                    holding_time_t += 1
                    if holding_time_t >= self.max_holding_time:
                        self.not_acceptable_count_trunk += 1
                        holding_time_t = 0
                elif alfa > TRUNK_INCL_THRESHOLD:
                    self.not_acceptable_count_trunk += 1
                else:
                    holding_time_t = 0

        bad_posture_percentage_r = (self.not_acceptable_shldr_r / time_r) * 100
        bad_posture_percentage_l = (self.not_acceptable_shldr_l / time_l) * 100
        bad_posture_percentage_t = (self.not_acceptable_count_trunk / time_t) * 100

        self.not_acceptable_percentages = bad_posture_percentage_r, bad_posture_percentage_l, bad_posture_percentage_t

    def plot_pie_charts(self):
        """
        Plot trunk and shoulders' posture pie charts with adjacent images.

        :return: fig (matplotlib.figure.Figure): The generated figure containing the plotted pie charts and images.
        """

        labels = ['Acceptable', 'Not acceptable']
        colors = ['green', 'red']

        plt.switch_backend('agg')

        fig, axs = plt.subplots(1, 6, figsize=(20, 4))
        plt.subplots_adjust(hspace=0.5, wspace=0.4)

        img_paths = [
            os.path.join(base_dir, 'img', 'left_shoulder.jpg'),
            os.path.join(base_dir, 'img', 'right_shoulder.jpg'),
            os.path.join(base_dir, 'img', 'trunk.jpg')
        ]

        for i in range(3):
            bad_percentage = self.not_acceptable_percentages[i]
            good_percentage = 100 - bad_percentage
            data = [good_percentage, bad_percentage]
            explode = (0, 0.1)

            wedges, _, autotexts = axs[2 * i].pie(data, explode=explode, shadow=True, colors=colors,
                                                  autopct='%1.1f%%', radius=1.2)

            for autotext in autotexts:
                autotext.set_fontsize(12)
                autotext.set_color('white')
                autotext.set_weight('bold')

            if i == 0:
                axs[2 * i].set_title('Left Shoulder Posture Risk', pad=18, fontsize=16)
            elif i == 1:
                axs[2 * i].set_title('Right Shoulder Posture Risk', pad=18, fontsize=16)
            else:
                axs[2 * i].set_title('Trunk Posture Risk', pad=15, fontsize=16)

            # Add image next to pie chart
            img = imread(img_paths[i])
            imagebox = OffsetImage(img, zoom=0.75)
            ab = AnnotationBbox(imagebox, (0.5, 0.5), frameon=False, xycoords='axes fraction',
                                boxcoords="axes fraction")
            axs[2 * i + 1].add_artist(ab)
            axs[2 * i + 1].axis('off')  # Hide axes for images

        fig.patch.set_facecolor('#ffffff')

        fig.legend(wedges, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=False,
                   fontsize=14)

        fig.tight_layout(rect=[0, 0, 0.9, 1])
        plt.close(fig)

        return fig

    def plot_angles_over_time(self):
        """
        Plot posture angles from data_old.json

        :return: fig (matplotlib.figure.Figure): The generated figure containing the plotted angles.
        """

        plt.switch_backend('agg')
        fig, axs = plt.subplots(3, 1, figsize=(8, 9))

        axs[0].plot(self.time, self.r_shldr, color='#38618C')
        axs[0].set_title('Right shoulder angle', fontsize=13)
        axs[0].grid(True)
        axs[0].set_ylim(top=180)

        axs[1].plot(self.time, self.l_shldr, color='#7E1946')
        axs[1].set_title('Left shoulder angle', fontsize=13)
        axs[1].set_ylabel('Angle (deg)')
        axs[1].set_ylim(top=180)
        axs[1].grid(True)

        axs[2].plot(self.time, self.trunk, color='#6abe40')
        axs[2].set_title('Trunk angle', fontsize=13)
        axs[2].grid(True)
        axs[2].set_ylim(top=180)
        axs[2].set_xlabel('Time (s)', fontsize=13)

        for ax in axs:
            ax.axhline(y=SHOULDERS_ANGLE_RANGE[0], color='green', linestyle='--')
            ax.axhline(y=SHOULDERS_ANGLE_RANGE[-1], color='red', linestyle='--')

        fig.patch.set_facecolor('#ffffff')
        fig.tight_layout()
        plt.close(fig)

        return fig, axs

    def plot_posture_status(self):
        """
        Plot posture status over time for right shoulder, left shoulder, and trunk.

        :return: fig (matplotlib.figure.Figure): The generated figure containing the plotted posture status.
        """

        plt.switch_backend('agg')
        fig, axs = plt.subplots(3, 1, figsize=(8, 9))

        y_right_arm = calculate_posture_status(self.r_shldr, SHOULDERS_ANGLE_THRESHOLD, SHOULDERS_ANGLE_RANGE)
        y_left_arm = calculate_posture_status(self.l_shldr, SHOULDERS_ANGLE_THRESHOLD, SHOULDERS_ANGLE_RANGE)
        y_trunk = calculate_posture_status(self.trunk, TRUNK_INCL_THRESHOLD, TRUNK_INCL_RANGE)
        y_labels = ['Acceptable', 'Checking', 'Not Acceptable']

        axs[0].plot(self.time, y_right_arm, color='#38618C')
        axs[0].set_title('Right Shoulder Posture Status', fontsize=13)
        axs[0].grid(True)
        axs[0].set_yticks([0, 1, 2])
        axs[0].set_yticklabels(y_labels)

        axs[1].plot(self.time, y_left_arm, color='#7E1946', label='Left Shoulder Posture Status')
        axs[1].set_title('Left Shoulder Posture Status', fontsize=13)
        axs[1].set_ylabel('Posture Status')
        axs[1].grid(True)
        axs[1].set_yticks([0, 1, 2])
        axs[1].set_yticklabels(y_labels)

        axs[2].plot(self.time, y_trunk, color='#6abe40', label='Trunk Posture Status')
        axs[2].set_title('Trunk Posture Status', fontsize=13)
        axs[2].grid(True)
        axs[2].set_xlabel('Time(s)', fontsize=13)
        axs[2].set_yticks([0, 1, 2])
        axs[2].set_yticklabels(y_labels)

        fig.patch.set_facecolor('#ffffff')
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_heatmap(self):
        """
        Plots a heatmap representing posture data over time.

        :return: fig (matplotlib.figure.Figure): The matplotlib Figure object containing the heatmap plot.
        """

        data = {
            'Time': pd.to_datetime(self.time, unit='s').strftime('%H:%M:%S'),
            'Left Shoulder Angle': self.l_shldr,
            'Right Shoulder Angle': self.r_shldr,
            'Trunk Inclination Angle': self.trunk
        }
        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(20, 4))
        sns.heatmap(df.set_index('Time').T, cmap='coolwarm', cbar=True, ax=ax)

        ax.set_xlabel('Time (h:m:s)')
        ax.set_title('Posture Heatmap Over Time')

        x_labels = ax.get_xticklabels()
        '''for i, label in enumerate(x_labels):
            if i % 30 != 0:
                label.set_visible(False)'''
        ax.set_xticklabels(x_labels)

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
        fig.tight_layout()
        fig.subplots_adjust(left=0.12, right=1)

        fig.patch.set_facecolor('#ffffff')
        plt.close(fig)
        return fig


