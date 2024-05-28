from src.PoseDetector import PoseDetector
from src.PostureEvaluation import PostureEvaluation
from src.config import data_path, MODEL_COMPLEXITY


class PoseManager:
    def __init__(self):
        self.pose = None
        self.posture = None
        self.recording = False

    def create_pose_detector(self):
        if self.pose is None:
            self.pose = PoseDetector(model_complexity=MODEL_COMPLEXITY)

    def destroy_pose_detector(self):
        if self.pose is not None:
            del self.pose
            self.pose = None

    def create_posture_eval(self):
        if self.posture is None:
            self.posture = PostureEvaluation(data_path=data_path)

    def destroy_posture_eval(self):
        if self.posture is not None:
            del self.posture
            self.posture = None

    def reset_posture_var(self):
        if self.posture is not None:
            self.posture.not_acceptable_shldr_r = 0
            self.posture.not_acceptable_shldr_l = 0
            self.posture.not_acceptable_count_trunk = 0
