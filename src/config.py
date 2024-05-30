import os

from mediapipe.python.solutions.pose import PoseLandmark

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
template_folder = os.path.join(base_dir, 'template')
data_path = os.path.join(base_dir, 'data')
model_path = os.path.join(base_dir, 'src', 'pose_landmarker.task')

MODEL_COMPLEXITY = 0
DEVICE_ID = 0

SHOULDERS_ANGLE_RANGE = range(21, 61)
SHOULDERS_ANGLE_THRESHOLD = 60

TRUNK_INCL_RANGE = range(21, 61)
TRUNK_INCL_THRESHOLD = 60

MAX_HOLDING_TIME = 20

RESIZE_PERCENT = 110

INTERVAL = 10

excluded_landmarks = [
    PoseLandmark.LEFT_EYE,
    PoseLandmark.RIGHT_EYE,
    PoseLandmark.LEFT_EYE_INNER,
    PoseLandmark.RIGHT_EYE_INNER,
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.NOSE,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT]

landmark_to_coords = {
    PoseLandmark.LEFT_ELBOW: 'left_elbow',
    PoseLandmark.LEFT_SHOULDER: 'left_shoulder',
    PoseLandmark.LEFT_HIP: 'left_hip',
    PoseLandmark.RIGHT_ELBOW: 'right_elbow',
    PoseLandmark.RIGHT_SHOULDER: 'right_shoulder',
    PoseLandmark.RIGHT_HIP: 'right_hip',
    PoseLandmark.LEFT_KNEE: 'left_knee',
    PoseLandmark.RIGHT_KNEE: 'right_knee',
}

landmark_connections = {
    PoseLandmark.LEFT_SHOULDER: [(12, 14), (12, 24)],
    PoseLandmark.RIGHT_SHOULDER: [(11, 13), (11, 23)],
    PoseLandmark.LEFT_HIP: [(11, 12), (12, 24), (11, 23), (23, 24)],
    PoseLandmark.RIGHT_HIP: [(11, 12), (12, 24), (11, 23), (23, 24)],
    PoseLandmark.LEFT_KNEE: [(11, 12), (12, 24), (11, 23), (23, 24)],
    PoseLandmark.LEFT_ELBOW: [(12, 14), (12, 24)],
    PoseLandmark.RIGHT_ELBOW: [(11, 13), (11, 23)]
}