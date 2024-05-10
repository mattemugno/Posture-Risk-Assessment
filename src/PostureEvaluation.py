import io
import json
import os
import seaborn as sns
from matplotlib import pyplot as plt

from src.config import MAX_HOLDING_TIME, ARM_ANGLE_RANGE, HEAD_ANGLE_RANGE, SHOULDERS_ANGLE_RANGE, TRUNK_INCL_RANGE
from src.utils import set_logger

set_logger()
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


### Trunk inclination
### Shoulder's angles
### Neck inclination ?

class PostureEvaluation:
    """
    This class is used to assess posture angles and show the risks associated with different parts of the body.
    """

    def __init__(self, data_path):
        with open(data_path, 'r') as file:
            data = json.load(file)
            self.time = [v['timestamp'] for v in data]
            self.r_shldr = [v['r_shldr'] for v in data]
            self.l_shldr = [v['l_shldr'] for v in data]
            self.trunk = [v['trunk_incl'] for v in data]

        self.max_holding_time = MAX_HOLDING_TIME
        self.arm_angle_range = ARM_ANGLE_RANGE
        self.head_angle_range = HEAD_ANGLE_RANGE
        self.shoulders_angle_range = SHOULDERS_ANGLE_RANGE
        self.trunk_incl_range = TRUNK_INCL_RANGE

        self.bad_percentages = None

        self.bad_posture_count_shldr_r = 0
        self.bad_posture_count_shldr_l = 0
        self.bad_posture_count_trunk = 0

    def calculate_percentages(self):
        """
        Calculate the percentage of time spent in a bad posture for each shoulder

        :return: Bad posture percentage based on each shoulder's angle
        """

        holding_time_r = 0
        holding_time_l = 0
        holding_time_t = 0
        previous_timestamp = self.time[0]

        for timestamp, r_shldr, l_shldr, trunk in zip(self.time, self.r_shldr, self.l_shldr, self.trunk):

            if not (self.arm_angle_range[0] <= r_shldr <= self.arm_angle_range[-1]):
                holding_time_r += timestamp - previous_timestamp
                if holding_time_r >= self.max_holding_time:
                    self.bad_posture_count_shldr_r += 1
                    holding_time_r = 0
            else:
                holding_time_r = 0

            if not (self.arm_angle_range[0] <= l_shldr <= self.arm_angle_range[-1]):
                holding_time_l += timestamp - previous_timestamp
                if holding_time_l >= self.max_holding_time:
                    self.bad_posture_count_shldr_l += 1
                    holding_time_l = 0
            else:
                holding_time_l = 0

            if not (self.trunk_incl_range[0] <= trunk <= self.trunk_incl_range[-1]):
                holding_time_t += timestamp - previous_timestamp
                if holding_time_t >= self.max_holding_time:
                    self.bad_posture_count_trunk += 1
                    holding_time_t = 0
            else:
                holding_time_t = 0

            previous_timestamp = timestamp

        total_time = self.time[-1] - self.time[0]
        bad_posture_percentage_r = (self.bad_posture_count_shldr_r / total_time) * 100
        bad_posture_percentage_l = (self.bad_posture_count_shldr_l / total_time) * 100
        bad_posture_percentage_t = (self.bad_posture_count_trunk / total_time) * 100

        self.bad_percentages = bad_posture_percentage_r, bad_posture_percentage_l, bad_posture_percentage_t

    def plot_pie_charts(self):
        labels = ['Acceptable', 'Not acceptable']
        colors = sns.color_palette('bright')

        plt.switch_backend('agg')

        # Create a figure and axes
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots
        plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Adjust spacing between subplots

        # Plot first two pie charts in the first row
        for i in range(2):
            bad_percentage = self.bad_percentages[i]
            good_percentage = 100 - bad_percentage
            data = [good_percentage, bad_percentage]
            explode = (0, 0.1)
            axs[0, i].pie(data, explode=explode, shadow=True, labels=labels, colors=colors, autopct='%1.1f%%',
                          radius=1.2)
            if i == 0:
                axs[0, i].set_title(f'Left Shoulder Posture Risk')
            else:
                axs[0, i].set_title(f'Right Shoulder Posture Risk')

        # Plot the third pie chart centered in the second row
        bad_percentage = self.bad_percentages[2]
        good_percentage = 100 - bad_percentage
        data = [good_percentage, bad_percentage]
        explode = (0, 0.1)
        axs[1, 0].pie(data, explode=explode, shadow=True, labels=labels, colors=colors, autopct='%1.1f%%', radius=1.2)
        axs[1, 0].set_title('Trunk posture Risk')

        # Hide the unused subplot in the second row (axs[1, 1])
        axs[1, 1].axis('off')

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()

        return img_buffer

