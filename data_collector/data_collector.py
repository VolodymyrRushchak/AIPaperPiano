import time
from random import randint, random

import cv2
import numpy as np
from PyQt5.QtCore import QTimerEvent
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import QWidget, QLabel, QCheckBox, QPushButton

from mylabel import MyLabel


class DataCollectorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowTitle("Data collector")
        self.setObjectName('datacollector')

        self.fps = 40
        self.startTimer(1000//self.fps)

        self.piano_keyboard = QLabel(self)
        self.piano_keyboard.setPixmap(QPixmap('assets/piano_keyboard.png').scaled(self.width(), 350))

        self.save_to_dataset_btn = QPushButton('Save to the dataset', self)
        self.save_to_dataset_btn.setGeometry(1000, 390, 350, 50)
        self.save_to_dataset_btn.setFont(QFont('serif', 16))
        self.save_to_dataset_btn.clicked.connect(self.save_to_dataset)

        self.ticks_before_saving = 2 * self.fps
        self.delay_counter = 0
        self.flash_ticks = 0.25 * self.fps
        self.flash_counter = 0

        self.random_setup_btn = QPushButton('Next random >', self)
        self.random_setup_btn.setGeometry(1000, 450, 350, 50)
        self.random_setup_btn.setFont(QFont('serif', 16))
        self.random_setup_btn.clicked.connect(self.random_setup)

        self.start_program_btn = QPushButton('Start program', self)
        self.start_program_btn.setGeometry(1000, 510, 350, 50)
        self.start_program_btn.setFont(QFont('serif', 16))
        self.start_program_btn.clicked.connect(self.start_program)
        self.program_interval = 0
        self.program_counter = 0
        self.white_keys_pattern = set()
        self.black_keys_pattern = set()
        self.toggle_control(False)

        self.reset_area_btn = QPushButton('Reset area', self)
        self.reset_area_btn.setGeometry(1000, 570, 350, 50)
        self.reset_area_btn.setFont(QFont('serif', 16))
        self.reset_area_btn.clicked.connect(self.reset_area)

        self.picture = MyLabel(self)
        self.picture.setGeometry(5, 355, 960, 540)
        self.picture.clicked.connect(self.unblock_control)

        self.vid = cv2.VideoCapture(0)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.pressed_keys = set()
        self.labels = []
        for i in range(51):
            label = QLabel(self)
            label.setStyleSheet('background-color: red')
            self.labels.append(label)

            check_box = QCheckBox(self)
            check_box.setObjectName(f'{i}')
            check_box.stateChanged.connect(lambda a, check_box=check_box: self.checkBoxHandler(check_box))

            if i < 30:
                label.setGeometry(11 + i*47, 220, 24, 80)
                check_box.setGeometry(15 + i*47, 300, 50, 50)
            else:
                label.setGeometry(36 + (i - 30) * 55 + ((i - 30) // 5) * 26 + ((i - 27) // 5) * 26, 50, 15, 80)
                check_box.setGeometry(35 + (i - 30) * 55 + ((i - 30) // 5) * 26 + ((i - 27) // 5) * 26, 120, 50, 50)

    def timerEvent(self, a0: QTimerEvent) -> None:
        ret, frame = self.vid.read()
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)\
            .scaled(self.picture.width(), self.picture.height())
        self.picture.setPixmap(QPixmap(img))

        if self.delay_counter != 0:
            self.delay_counter += 1
            self.save_to_dataset_btn.setText('Save to the dataset' +
                                             f' {3-3*self.delay_counter//self.ticks_before_saving}')
        if self.delay_counter == self.ticks_before_saving:
            self.save_to_dataset_btn.setText('Save to the dataset')
            self.save_to_dataset()

        if self.flash_counter != 0:
            self.flash_counter += 1
            self.setStyleSheet('QWidget#datacollector {background-color: #f2f542}')
        if self.flash_counter == self.flash_ticks:
            self.flash_counter = 0
            self.setStyleSheet('QWidget#datacollector {background-color: #f0f0f0}')

        if self.program_counter != 0:
            self.update_program()

    def checkBoxHandler(self, checkbox: QCheckBox) -> None:
        index = int(checkbox.objectName())
        if checkbox.isChecked():
            self.press_key(index)
        else:
            self.release_key(index)

    def press_key(self, index: int) -> None:
        if index < 0 or index > 50:
            raise ValueError()
        self.pressed_keys.add(index)
        self.labels[index].setStyleSheet('background-color: #5afc03')

    def release_key(self, index: int) -> None:
        if index < 0 or index > 50:
            raise ValueError()
        if index in self.pressed_keys:
            self.pressed_keys.remove(index)
        self.labels[index].setStyleSheet('background-color: red')

    def save_to_dataset(self) -> None:
        if self.delay_counter == 0:
            self.delay_counter = 1
        else:
            self.save_img()
            self.delay_counter = 0

    def save_img(self) -> None:
        pressed_keys = '_'.join(map(str, self.pressed_keys))
        piano_area = '_'.join(map(lambda qpoint: f"{qpoint.x()}_{qpoint.y()}", self.picture.piano_boundary))
        self.picture.pixmap().save(f'dataset/{pressed_keys}#{piano_area}@{time.time_ns()}.png', 'png')
        self.flash_counter = 1

    def random_setup(self) -> None:
        self.reset_keyboard()
        # pressed_num = randint(1, 6)
        # for i in range(pressed_num):
        #     index = randint(0, 50)
        #     while index in self.pressed_keys:
        #         index = randint(0, 50)
        #     self.press_key(index)
        white_pressed_num = randint(1, 5)
        black_pressed_num = 1
        for i in range(white_pressed_num):
            white_index = self.get_white_index()
            while white_index in self.pressed_keys:
                white_index = self.get_white_index()
            self.press_key(white_index)
        for i in range(black_pressed_num):
            black_index = self.get_black_index()
            while black_index in self.pressed_keys:
                black_index = self.get_black_index()
            self.press_key(black_index)

    def get_white_index(self):
        return np.random.choice(range(30), p=[
             0.05973451, 0.05973451, 0.03982301, 0.03539823, 0.01769912, 0.01327434,
             0.00663717, 0.00442478, 0., 0., 0., 0.,
             0., 0., 0., 0.01106195, 0.00884956, 0.02212389,
             0.03097345, 0.02876106, 0.03761062, 0.03761062, 0.05973451, 0.0619469,
             0.05530973, 0.06415929, 0.07743363, 0.08185841, 0.08628319, 0.09955752])

    def get_black_index(self):
        return np.random.choice(range(30, 51), p=[
            0.1036036 , 0.10810811, 0.0990991 , 0.0990991 , 0.07207207, 0.05855856,
            0.05405405, 0.04954955, 0.03603604, 0.01351351, 0.03153153, 0.01801802,
            0.01801802, 0.01351351, 0.01351351, 0.01801802, 0.03603604, 0.03153153,
            0.03603604, 0.04054054, 0.04954955])

    def start_program(self) -> None:
        self.reset_keyboard()
        pressed_num = randint(4, 7)
        self.press_key(0)
        self.white_keys_pattern.add(0)
        key_range = 13 - pressed_num
        sample = np.random.choice(key_range, pressed_num - 1) - key_range
        for index in sample:
            r = random()
            if r < 0.2:
                self.white_keys_pattern.add(index)
            else:
                self.black_keys_pattern.add(index + 30)
        self.program_interval = self.fps * 2
        self.program_counter = 1

    def update_program(self) -> None:
        self.program_counter += 1
        self.start_program_btn.setText(f"Start program {5 - 5 * self.program_counter // self.program_interval}")
        if self.program_counter >= self.program_interval:
            self.save_img()
            white_copy, black_copy = self.white_keys_pattern.copy(), self.black_keys_pattern.copy()
            self.reset_keyboard()
            for index in white_copy:
                new_index = index + 1
                self.white_keys_pattern.add(new_index)
                if 0 <= new_index < 30:
                    self.press_key(new_index)
            for index in black_copy:
                new_index = index + 1
                self.black_keys_pattern.add(new_index)
                if 30 <= new_index < 51:
                    self.press_key(new_index)
            if not self.pressed_keys:
                self.program_counter = 0
                return
            self.program_interval = self.fps * len(self.pressed_keys) * 1.2
            self.program_counter = 1

    def reset_keyboard(self) -> None:
        self.pressed_keys.clear()
        self.white_keys_pattern.clear()
        self.black_keys_pattern.clear()
        for label in self.labels:
            label.setStyleSheet('background-color: red')

    def unblock_control(self) -> None:
        if len(self.picture.piano_boundary) == 4:
            self.toggle_control(True)

    def toggle_control(self, on: bool):
        self.start_program_btn.setVisible(on)
        self.save_to_dataset_btn.setVisible(on)

    def reset_area(self) -> None:
        self.picture.piano_boundary = []
        self.toggle_control(False)
