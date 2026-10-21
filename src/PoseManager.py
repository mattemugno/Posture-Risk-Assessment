import logging
from threading import Event, RLock, Thread

from src.PoseDetector import PoseDetector
from src.PostureEvaluation import PostureEvaluation
from src.config import MODEL_COMPLEXITY


class PoseManager:
    def __init__(self):
        self.pose = None
        self.posture = None
        self.recording = False
        self.lock = RLock()
        self.posture_lock = RLock()
        self.frame = None
        self._capture_stop = None
        self._capture_thread = None

    def create_pose_detector(self):
        with self.lock:
            if self.pose is None:
                self.pose = PoseDetector(model_complexity=MODEL_COMPLEXITY)
                self.frame = None
                self.recording = True
                self._capture_stop = Event()
                self._capture_thread = Thread(target=self._capture_frames,
                                              args=(self.pose, self._capture_stop), daemon=True)
                self._capture_thread.start()

    def _capture_frames(self, pose, stop):
        while not stop.is_set():
            with self.lock:
                if stop.is_set():
                    return
                try:
                    self.frame = pose.find_landmarks()
                except Exception:
                    logging.exception('Unable to read camera frame.')
                    self.frame = None
                missing_frame = self.frame is None
            # Failed reads must not spin; successful reads are paced by the camera.
            stop.wait(0.1 if missing_frame else 0.001)

    def get_frame(self):
        with self.lock:
            return self.frame

    def destroy_pose_detector(self):
        with self.lock:
            if self.pose is not None:
                if self.pose.record_session:
                    self.pose.save_data()
                    self.pose.record_session = False
                self._capture_stop.set()
                try:
                    self.pose.close()
                finally:
                    self.pose = None
                    self.frame = None
                    self.recording = False

    def create_posture_eval(self, filename):
        with self.posture_lock:
            # Replace only after loading succeeds, retaining the previous selection on error.
            self.posture = PostureEvaluation(filename=filename)

    def destroy_posture_eval(self):
        with self.posture_lock:
            if self.posture is not None:
                del self.posture
                self.posture = None

    def reset_posture_var(self):
        with self.posture_lock:
            if self.posture is not None:
                self.posture.not_acceptable_shldr_r = 0
                self.posture.not_acceptable_shldr_l = 0
                self.posture.not_acceptable_count_trunk = 0
