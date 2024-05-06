import cv2


class VideoCamera(object):
    """
    This class handles camera methods
    """

    def __init__(self, device_id=0):
        """
        Initialize the VideoCamera instance.

        :param device_id: int, optional
            Camera id. Default is 0, which usually refers to the built-in webcam.
        """
        self.video = cv2.VideoCapture(device_id)

    def __del__(self):
        """
        Destructor method to release video resources and close windows held by the object instance.

        :return: None
        """

        self.video.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        """
        Retrieve a frame from the video camera.

        :return: A tuple (success, image) where:
                 - success (bool): True if a frame was successfully retrieved from the camera.
                 - image (numpy.ndarray): The RGB image array representing the retrieved frame.
        """

        success, frame = self.video.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        return success, image

    def isOpen(self):
        """
        Check if the camera device is active.

        :return:
        """
        return self.video.isOpened()
