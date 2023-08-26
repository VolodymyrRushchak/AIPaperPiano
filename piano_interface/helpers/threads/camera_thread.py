import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from helpers.constants import WEBCAM_WIDTH, WEBCAM_HEIGHT


class CameraThread(QThread):
    image = pyqtSignal(np.ndarray)

    def __init__(self, prediction_thread):
        super().__init__()
        self.capture = None
        self.prediction_thread = prediction_thread

    def start_capture(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

    def stop_capture(self):
        if self.capture:
            self.capture.release()
            self.capture = None

    def run(self):
        self.start_capture()
        while True:
            ret, frame = self.capture.read()
            if ret:
                self.prediction_thread.set_image(frame)
                self.image.emit(frame)

    def stop(self):
        self.stop_capture()
        super().stop()
