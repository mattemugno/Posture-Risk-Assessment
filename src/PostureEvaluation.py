import io
import json
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

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
        Plot trunk and shoulders' posture pie charts.

        :return: fig (matplotlib.figure.Figure): The generated figure containing the plotted pie charts.
        """

        labels = ['Acceptable', 'Not acceptable']
        colors = sns.color_palette('bright')

        plt.switch_backend('agg')

        fig, axs = plt.subplots(1, 3, figsize=(15, 3))
        plt.subplots_adjust(hspace=0.5, wspace=0.4)

        for i in range(3):
            bad_percentage = self.not_acceptable_percentages[i]
            good_percentage = 100 - bad_percentage
            data = [good_percentage, bad_percentage]
            explode = (0, 0.1)

            _, texts, _ = axs[i].pie(data, explode=explode, shadow=True, labels=labels, colors=colors,
                                     autopct='%1.1f%%',
                                     radius=1.2)

            texts[0].set_fontsize(11)
            texts[1].set_fontsize(11)

            if i == 0:
                axs[i].set_title(f'Left Shoulder Posture Risk', pad=15)
            elif i == 1:
                axs[i].set_title(f'Right Shoulder Posture Risk', pad=15)
            else:
                axs[i].set_title('Trunk posture Risk', pad=15)

        plt.close(fig)

        return fig

    def plot_angles_over_time(self):
        """
        Plot posture angles from data.json

        :return: fig (matplotlib.figure.Figure): The generated figure containing the plotted angles.
        """

        plt.switch_backend('agg')
        fig, axs = plt.subplots(1, 3, sharex=True, figsize=(15, 3), constrained_layout=True)

        axs[0].plot(self.time, self.r_shldr, color='blue')
        axs[0].set_title('Right shoulder angle', fontsize=11)
        axs[0].grid(True)
        axs[0].set_ylabel('Angle (deg)')
        axs[0].set_ylim(top=180)

        axs[1].plot(self.time, self.l_shldr, color='green')
        axs[1].set_title('Left shoulder angle', fontsize=11)
        axs[1].grid(True)
        axs[1].get_yaxis().set_ticklabels([])

        axs[2].plot(self.time, self.trunk, color='red')
        axs[2].set_title('Trunk angle', fontsize=11)
        axs[2].grid(True)
        axs[1].get_yaxis().set_ticklabels([])

        for ax in axs:
            ax.set_xlabel('Time (s)', fontsize=11)

        plt.close(fig)

        return fig

    def plot_posture_status(self):
        """
        Plot posture status over time for right shoulder, left shoulder, and trunk.

        :return: fig (matplotlib.figure.Figure): The generated figure containing the plotted posture status.
        """

        plt.switch_backend('agg')
        fig, axs = plt.subplots(1, 3, figsize=(15, 3), constrained_layout=True)

        y_right_arm = calculate_posture_status(self.r_shldr, SHOULDERS_ANGLE_THRESHOLD, SHOULDERS_ANGLE_RANGE)
        y_left_arm = calculate_posture_status(self.l_shldr, SHOULDERS_ANGLE_THRESHOLD, SHOULDERS_ANGLE_RANGE)
        y_trunk = calculate_posture_status(self.trunk, TRUNK_INCL_THRESHOLD, TRUNK_INCL_RANGE)
        y_labels = ['Acceptable', 'Checking', 'Not Acceptable']

        axs[0].plot(self.time, y_right_arm, color='blue')
        axs[0].set_title('Right Shoulder Posture Status', fontsize=11)
        axs[0].grid(True)
        axs[0].tick_params(axis='y', rotation=30)
        axs[0].set_yticks([0, 1, 2])
        axs[0].set_yticklabels(y_labels)
        axs[0].set_ylabel('Posture Status', fontsize=11)

        axs[1].plot(self.time, y_left_arm, color='green', label='Left Shoulder Posture Status')
        axs[1].set_title('Left Shoulder Posture Status', fontsize=11)
        axs[1].grid(True)
        axs[1].get_yaxis().set_ticklabels([])

        axs[2].plot(self.time, y_trunk, color='red', label='Trunk Posture Status')
        axs[2].set_title('Trunk Posture Status', fontsize=11)
        axs[2].grid(True)
        axs[2].get_yaxis().set_ticklabels([])

        for ax in axs:
            ax.set_xlabel('Time(s)', fontsize=11)
            ax.set_ylim([-0.5, 3])

        plt.close(fig)
        return fig
