import cv2


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
        self.recording = True

    def __del__(self):
        """
        Destructor method to release video resources and close windows held by the object instance.

        :return: None
        """

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

        if not success:
            return None, None

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
