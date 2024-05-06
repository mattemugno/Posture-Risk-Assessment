import json
import os

from src.config import MAX_HOLDING_TIME, ARM_ANGLE_RANGE, HEAD_ANGLE_RANGE, SHOULDERS_ANGLE_RANGE
from src.utils import set_logger

set_logger()
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


### Trunk inclination
### Shoulder's angles
### Neck inclination ?
### Shoulders inclination


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
            # self.trunk = [v['trunk_incl'] for v in data]

        self.max_holding_time = MAX_HOLDING_TIME
        self.arm_angle_range = ARM_ANGLE_RANGE
        self.head_angle_range = HEAD_ANGLE_RANGE
        self.shoulders_angle_range = SHOULDERS_ANGLE_RANGE

        self.bad_posture_count_shldr_r = 0
        self.bad_posture_count_shldr_l = 0

    def get_percentage_shoulders(self):
        """
        Calculate the percentage of time spent in a bad posture for each shoulder

        :return: Bad posture percentage based on each shoulder's angle
        """

        holding_time_r = 0
        holding_time_l = 0
        previous_timestamp = self.time[0]

        for timestamp, r_shldr, l_shldr in zip(self.time, self.r_shldr, self.l_shldr):
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

            previous_timestamp = timestamp

        total_time = self.time[-1] - self.time[0]
        bad_posture_percentage_r = (self.bad_posture_count_shldr_r / total_time) * 100
        bad_posture_percentage_l = (self.bad_posture_count_shldr_l / total_time) * 100

        return bad_posture_percentage_r, bad_posture_percentage_l
