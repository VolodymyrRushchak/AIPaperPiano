from typing import List

import cv2
import numpy as np
import tensorflow as tf

from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton
from PyQt5.QtMultimedia import QSound

from helpers.constants import WEBCAM_WIDTH
from helpers.threads.prediction_thread import PredictionThread
from helpers.ui_elements.camera_label import CameraLabel
from helpers.threads.camera_thread import CameraThread
from helpers.ui_elements.clickable_label import ClickableLabel
from helpers.threads.mythread import MyThread
from helpers.ai_model import get_ai_model


class PianoWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.indicators = []
        white_notes = ['C2', 'D2', 'E2', 'F2', 'G2',
                       'A2', 'B2', 'C3', 'D3', 'E3', 'F3', 'G3',
                       'A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4',
                       'A4', 'B4', 'C5', 'D5', 'E5', 'F5', 'G5',
                       'A5', 'B5', 'C6', 'D6']
        black_notes = ['Db2', 'Eb2', 'Gb2', 'Ab2',
                       'Bb2', 'Db3', 'Eb3', 'Gb3', 'Ab3',
                       'Bb3', 'Db4', 'Eb4', 'Gb4', 'Ab4',
                       'Bb4', 'Db5', 'Eb5', 'Gb5', 'Ab5',
                       'Bb5', 'Db6']
        self.notes = white_notes + black_notes
        self.sounds = [QSound(f"assets/notes/{note}.wav") for note in self.notes]
        self.pressed_keys = np.zeros(51)

        self.setupUi()

        self.aimodel = get_ai_model()
        self.prediction_thread = PredictionThread(
            self.picture.piano_boundary, self.aimodel, WEBCAM_WIDTH / self.picture.width())
        self.prediction_thread.prediction.connect(self.handle_predictions)
        self.prediction_thread.start()

        self.camera = CameraThread(self.prediction_thread)
        self.camera.image.connect(self.update_image)
        self.camera.start()

    def setupUi(self) -> None:
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowTitle("Paper Piano")
        self.piano_keyboard = QLabel(self)
        self.piano_keyboard.setPixmap(QPixmap('assets/piano_keyboard.png').scaled(self.width(), 350))
        self.picture = CameraLabel(self)
        self.picture.setGeometry(self.width() // 2 - 960 // 2, 355, 960, 540)
        self.picture.clicked.connect(lambda e: self.picture.piano_boundary.append(QPoint(e.x(), e.y())) if len(
            self.picture.piano_boundary) < 4 else None)
        for i in range(51):
            label = ClickableLabel(self)
            label.setStyleSheet('background-color: red')
            label.clicked.connect(lambda ev, note_idx=i: self.play_notes([note_idx]))
            self.indicators.append(label)
            if i < 30:
                label.setGeometry(11 + i * 47, 220, 24, 80)
            else:
                label.setGeometry(36 + (i - 30) * 55 + ((i - 30) // 5) * 26 + ((i - 27) // 5) * 26, 50, 15, 80)
        self.reset_area_btn = QPushButton('Reset', self)
        self.reset_area_btn.setGeometry(1200, 370, 165, 50)
        self.reset_area_btn.setFont(QFont('serif', 16))
        self.reset_area_btn.clicked.connect(lambda: self.picture.piano_boundary.clear())

    def handle_predictions(self, predictions: np.ndarray) -> None:
        predictions = tf.cast(tf.squeeze(tf.sigmoid(predictions)) > 0.1, dtype=tf.float32)
        notes_to_play = [i for i, p in enumerate(predictions) if p == 1 and self.pressed_keys[i] == 0]
        self.pressed_keys = predictions
        self.play_notes(notes_to_play)

    def update_image(self, frame: np.ndarray) -> None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.picture.width(), self.picture.height()))
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.picture.setPixmap(QPixmap.fromImage(image))

    def play_notes(self, note_indices: List[int]) -> None:
        for note_idx in note_indices:
            self.sounds[note_idx].play()
            self.indicators[note_idx].setStyleSheet('background-color: #5afc03')
            thread = MyThread(self, self.sounds[note_idx])
            thread.finished.connect(lambda note_idx=note_idx:
                                    self.indicators[note_idx].setStyleSheet('background-color: red'))
            thread.start()
