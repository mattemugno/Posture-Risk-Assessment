import io
import json
import os
import seaborn as sns
from matplotlib import pyplot as plt

from src.config import MAX_HOLDING_TIME, SHOULDERS_ANGLE_RANGE, TRUNK_INCL_RANGE, \
    TRUNK_INCL_THRESHOLD, SHOULDERS_ANGLE_THRESHOLD
from src.utils import set_logger

set_logger()
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


### Trunk inclination -- alfa angle
### Shoulder's angles

class PostureEvaluation:
    """
    This class is used to assess posture angles and show the risks associated with different parts of the body.
    """

    def __init__(self, data_path):
        self.data_path = data_path

        with open(self.data_path, 'r') as file:
            data = json.load(file)
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
        previous_timestamp = self.time[0]

        for timestamp, gamma_r, gamma_l, alfa in zip(self.time, self.r_shldr, self.l_shldr, self.trunk):

            # 0 < gamma < 20 acceptable
            # 20 < gamma < 60 check holding time
            # gamma > 60 not acceptable
            if gamma_r is not None:
                if not (self.shoulders_angle_range[0] <= gamma_r <= self.shoulders_angle_range[-1]):
                    holding_time_r += timestamp - previous_timestamp
                    if holding_time_r >= self.max_holding_time:
                        self.not_acceptable_shldr_r += 1
                        holding_time_r = 0
                elif gamma_r > SHOULDERS_ANGLE_THRESHOLD:
                    self.not_acceptable_shldr_r += 1
                else:
                    holding_time_r = 0

            if gamma_l is not None:
                if not (self.shoulders_angle_range[0] <= gamma_l <= self.shoulders_angle_range[-1]):
                    holding_time_l += timestamp - previous_timestamp
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
                if not (self.trunk_incl_range[0] <= alfa <= self.trunk_incl_range[-1]):
                    holding_time_t += timestamp - previous_timestamp
                    if holding_time_t >= self.max_holding_time:
                        self.not_acceptable_count_trunk += 1
                        holding_time_t = 0
                elif self.trunk > TRUNK_INCL_THRESHOLD:
                    self.not_acceptable_count_trunk += 1
                else:
                    holding_time_t = 0

            previous_timestamp = timestamp

        total_time = self.time[-1] - self.time[0]
        bad_posture_percentage_r = (self.not_acceptable_shldr_r / total_time) * 100 if total_time > 0 else 0
        bad_posture_percentage_l = (self.not_acceptable_shldr_l / total_time) * 100 if total_time > 0 else 0
        bad_posture_percentage_t = (self.not_acceptable_count_trunk / total_time) * 100 if total_time > 0 else 0

        self.not_acceptable_percentages = bad_posture_percentage_r, bad_posture_percentage_l, bad_posture_percentage_t

    def plot_pie_charts(self):
        """
        Plot trunk and shoulders' posture pie charts.
        :return:
        """
        labels = ['Acceptable', 'Not acceptable']
        colors = sns.color_palette('bright')

        plt.switch_backend('agg')

        # Create a figure and axes
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots
        plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Adjust spacing between subplots

        # Plot first two pie charts in the first row
        for i in range(2):
            bad_percentage = self.not_acceptable_percentages[i]
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
        bad_percentage = self.not_acceptable_percentages[2]
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

    def plot_posture(self):
        """
        Plot posture angles from data.json

        :return: BytesIO object containing the plot image
        """
        with open(self.data_path, 'r') as file:
            data = json.load(file)
            data = [v for v in data if v is not None]
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
