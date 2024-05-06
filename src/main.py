import os

from src.PoseDetector import PoseDetector
from src.PostureEvaluation import PostureEvaluation
from src.utils import plot_pie_charts_shoulders

device_id = 0
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


if __name__ == '__main__':
    pose = PoseDetector(enable_face_landmarks=False)
    pose.find_landmarks(device_id)
    # pose.save_data()
    # pose.plot_posture()
    eval_posture = PostureEvaluation(pose.data_path)
    plot_pie_charts_shoulders(eval_posture.get_percentage_shoulders())
