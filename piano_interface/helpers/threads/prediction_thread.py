import time
from typing import List

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QPoint
import tensorflow as tf

from helpers.preprocessing import preprocess


class PredictionThread(QThread):
    prediction = pyqtSignal(np.ndarray)

    def __init__(self, piano_boundary: List[QPoint], aimodel: tf.keras.Model, scaling_factor: float):
        super().__init__()
        self.piano_boundary = piano_boundary
        self.aimodel = aimodel
        self.scaling_factor = scaling_factor

        self.image = None

    def set_image(self, image: np.ndarray) -> None:
        self.image = image

    def run(self):
        while True:
            if len(self.piano_boundary) == 4 and self.image is not None:
                predictions = self.get_predictions()
                self.prediction.emit(predictions)
            else:
                time.sleep(0.005)

    def get_predictions(self) -> np.ndarray:
        piano_boundary = tf.constant([[int(point.x() * self.scaling_factor), int(point.y() * self.scaling_factor)]
                                      for point in self.piano_boundary])
        img = tf.convert_to_tensor(self.image)
        img = preprocess(img, piano_boundary)
        # t = time.perf_counter()
        preds = self.aimodel.predict(img, verbose=0)
        # print(time.perf_counter() - t)
        return preds
