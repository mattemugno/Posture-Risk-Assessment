import json
import logging
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.image import imread
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from src.config import MAX_HOLDING_TIME, SHOULDERS_ANGLE_RANGE, TRUNK_INCL_RANGE, \
    TRUNK_INCL_THRESHOLD, SHOULDERS_ANGLE_THRESHOLD, base_dir, data_path, INTERVAL
from src.utils import calculate_posture_status
from src.plot_style import ANGLE_CMAP, GREEN, INK, MUTED, ROSE, SERIES_COLORS, \
    finish_timeline, style_axis


### Trunk inclination -- alfa angle
### Shoulder's angles

class PostureEvaluation:
    """
    This class is used to assess posture angles and show the risks associated with different parts of the body.
    """

    def __init__(self, filename):
        """
        Initializes the PostureEvaluation instance
        :arg data_path: Path of the json file
        """

        self.data_path = os.path.realpath(os.path.join(data_path, filename))
        if (os.path.dirname(self.data_path) != os.path.realpath(data_path)
                or not self.data_path.endswith('.json')):
            raise ValueError('Select a JSON file from the data directory.')

        plt.set_loglevel('WARNING')

        with open(self.data_path, 'r') as file:
            data = json.load(file)
            data = [v for v in data if
                    v['r_shldr'] is not None and v['l_shldr'] is not None and v['trunk_incl'] is not None]
            self.time = [v['timestamp'] for v in data]
            self.r_shldr = [v['r_shldr'] for v in data]
            self.l_shldr = [v['l_shldr'] for v in data]
            self.trunk = [v['trunk_incl'] for v in data]
        logging.info('Loaded %d samples from %s', len(data), self.data_path)

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
        """Show the existing risk percentages as three annotated doughnuts."""
        labels = ['Acceptable', 'Not acceptable']
        plt.switch_backend('agg')
        fig, axs = plt.subplots(
            1, 6, figsize=(12.6, 3.5), dpi=150,
            gridspec_kw={'width_ratios': [1.6, 0.5, 1.6, 0.5, 1.6, 0.5]}
        )
        img_paths = [
            os.path.join(base_dir, 'img', 'left_shoulder.jpg'),
            os.path.join(base_dir, 'img', 'right_shoulder.jpg'),
            os.path.join(base_dir, 'img', 'trunk.jpg')
        ]
        titles = ['Left shoulder', 'Right shoulder', 'Trunk']

        for i in range(3):
            bad_percentage = self.not_acceptable_percentages[i]
            good_percentage = 100 - bad_percentage
            data = [good_percentage, bad_percentage]
            ax = axs[2 * i]
            wedges, _ = ax.pie(
                data, colors=[GREEN, ROSE], startangle=90, counterclock=False,
                wedgeprops={'width': 0.18, 'edgecolor': 'white', 'linewidth': 2}
            )
            ax.set_title(titles[i], pad=16, fontsize=11, fontweight='medium', color=INK)
            ax.text(0, 0.06, f'{bad_percentage:.1f}%', ha='center', va='center',
                    fontsize=23, fontweight='medium', color=INK)
            ax.text(0, -0.24, 'not acceptable', ha='center', fontsize=8, color=MUTED)
            ax.text(0, -1.27, f'{good_percentage:.1f}% acceptable', ha='center',
                    fontsize=9, color=GREEN)

            img = imread(img_paths[i])
            imagebox = OffsetImage(img, zoom=140 / max(img.shape[:2]), dpi_cor=False)
            ab = AnnotationBbox(imagebox, (0.5, 0.5), frameon=False,
                               xycoords='axes fraction', boxcoords='axes fraction')
            axs[2 * i + 1].add_artist(ab)
            axs[2 * i + 1].axis('off')

        fig.set_facecolor('white')
        fig.legend(wedges, labels, loc='lower center', ncol=2, frameon=False,
                   fontsize=9, labelcolor=MUTED, bbox_to_anchor=(0.5, 0.005),
                   handlelength=1.1, columnspacing=2.5)
        fig.subplots_adjust(left=0.025, right=0.98, top=0.8, bottom=0.2, wspace=0.18)
        plt.close(fig)
        return fig

    def plot_angles_over_time(self):
        """Plot the original angle samples and threshold lines."""
        plt.switch_backend('agg')
        fig, axs = plt.subplots(3, 1, figsize=(6.8, 8.2), dpi=150)
        time_labels = pd.to_datetime(self.time, unit='s').strftime('%H:%M:%S')
        series = [self.r_shldr, self.l_shldr, self.trunk]
        titles = ['Right shoulder angle', 'Left shoulder angle', 'Trunk angle']

        for ax, values, title, color in zip(axs, series, titles, SERIES_COLORS):
            line, = ax.plot(time_labels, values, color=color, linewidth=1.7)
            ax.fill_between(line.get_xdata(orig=False), values, color=color, alpha=0.08)
            ax.set_title(title, loc='left', fontsize=11, fontweight='medium', color=INK, pad=15)
            ax.set_ylim(top=180)
            ax.axhline(y=SHOULDERS_ANGLE_RANGE[0], color=GREEN,
                       linestyle=(0, (4, 4)), linewidth=0.8, alpha=0.55)
            ax.axhline(y=SHOULDERS_ANGLE_RANGE[-1], color=ROSE,
                       linestyle=(0, (4, 4)), linewidth=0.8, alpha=0.55)

        axs[1].set_ylabel('Angle (deg)', labelpad=12)
        axs[2].set_xlabel('Time (H:M:S)', labelpad=12)
        finish_timeline(fig, axs, time_labels, INTERVAL)
        plt.close(fig)
        return fig, axs

    def plot_posture_status(self):
        """Plot the existing status calculations with the shared chart styling."""
        plt.switch_backend('agg')
        fig, axs = plt.subplots(3, 1, figsize=(6.8, 8.2), dpi=150)
        time_labels = pd.to_datetime(self.time, unit='s').strftime('%H:%M:%S')
        y_right_arm = calculate_posture_status(self.r_shldr, SHOULDERS_ANGLE_THRESHOLD, SHOULDERS_ANGLE_RANGE)
        y_left_arm = calculate_posture_status(self.l_shldr, SHOULDERS_ANGLE_THRESHOLD, SHOULDERS_ANGLE_RANGE)
        y_trunk = calculate_posture_status(self.trunk, TRUNK_INCL_THRESHOLD, TRUNK_INCL_RANGE)
        y_labels = ['Acceptable', 'Checking', 'Not Acceptable']
        titles = ['Right shoulder status', 'Left shoulder status', 'Trunk status']

        for ax, values, title, color in zip(
                axs, [y_right_arm, y_left_arm, y_trunk], titles, SERIES_COLORS):
            line, = ax.plot(time_labels, values, color=color, linewidth=1.6)
            ax.fill_between(line.get_xdata(orig=False), values, color=color, alpha=0.06)
            ax.set_title(title, loc='left', fontsize=11, fontweight='medium', color=INK, pad=15)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(y_labels)
            ax.set_ylim(-0.18, 2.18)

        axs[1].set_ylabel('Posture status', labelpad=12)
        axs[2].set_xlabel('Time (H:M:S)', labelpad=12)
        finish_timeline(fig, axs, time_labels, INTERVAL, left=0.26)
        plt.close(fig)
        return fig

    def plot_heatmap(self):
        """Render the original angle matrix with a softer sequential palette."""
        data = {
            'Time': pd.to_datetime(self.time, unit='s').strftime('%H:%M:%S'),
            'Left Shoulder Angle': self.l_shldr,
            'Right Shoulder Angle': self.r_shldr,
            'Trunk Inclination Angle': self.trunk
        }
        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(12.4, 3.2), dpi=150)
        sns.heatmap(
            df.set_index('Time').T, cmap=ANGLE_CMAP, cbar=True, ax=ax,
            xticklabels=max(1, (len(df) + 7) // 8),
            cbar_kws={'label': 'Angle (deg)', 'shrink': 0.65, 'pad': 0.025}
        )
        style_axis(ax)
        ax.grid(False)
        ax.set_xlabel('Time (H:M:S)', labelpad=12)
        ax.set_ylabel('')
        ax.set_title('Posture heatmap over time', loc='left', pad=18,
                     fontsize=11, fontweight='medium', color=INK)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=9, colors=MUTED, length=0)
        colorbar.ax.yaxis.label.set(color=MUTED, fontsize=9)
        fig.set_facecolor('white')
        fig.subplots_adjust(left=0.19, right=0.98, bottom=0.25, top=0.8)
        plt.close(fig)
        return fig
