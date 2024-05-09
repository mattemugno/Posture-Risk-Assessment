from mediapipe.python.solutions.pose import PoseLandmark

ARM_ANGLE_THRESHOLD = 45
ARM_ANGLE_RANGE = range(20, 60)
HEAD_ANGLE_RANGE = range(25, 85)
SHOULDERS_ANGLE_RANGE = range(20, 60)
TRUNK_INCL_RANGE = range(20, 60)
MAX_HOLDING_TIME = 4

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
    PoseLandmark.LEFT_KNEE: 'left_knee'
}