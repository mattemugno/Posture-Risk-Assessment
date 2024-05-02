import os

from src.PoseDetector import PoseDetector

device_id = 0
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

if __name__ == "__main__":
    pose = PoseDetector()
    pose.find_landmarks(device_id)
    pose.write_data()
    pose.plot_posture()



