import logging

import cv2

from src.config import RESIZE_PERCENT
from src.utils import rescale_frame


class VideoCamera(object):
    """
    This class handles camera methods
    """

    def __init__(self, source=0):
        """
        Initialize the VideoCamera instance.

        :param source: int, optional
            Camera id. Default is 0, which usually refers to the built-in webcam.
        """
        self.video = cv2.VideoCapture(source)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        self.recording = True

    def __del__(self):
        """
        Destructor method to release video resources and close windows held by the object instance.

        :return: None
        """

        logging.info('Release Cam')
        self.close()

    def close(self):
        self.video.release()

    def get_frame(self, holistic):
        """
        Retrieve a frame from the video camera and process it using the specified holistic detector.

        :param holistic: The Holistic object that processes the frame.
        :return: A tuple (image, results) where:
                 - image (numpy.ndarray): The BGR image array representing the retrieved frame.
                 - results: The detection results from the holistic detector.
        """

        success, frame = self.video.read()

        if not success or frame is None:
            return None, None

        if frame is not None:
            frame = rescale_frame(frame, percent=RESIZE_PERCENT)

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = holistic.process(image)

        # Recolor image to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image, results

    def isOpen(self):
        """
        Check if the camera device is active.

        :return:
        """
        return self.video.isOpened()
