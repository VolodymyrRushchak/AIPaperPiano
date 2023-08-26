from PyQt5 import QtCore
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import QWidget


class MyThread(QtCore.QThread):
    def __init__(self, parent: QWidget, sound: QSound):
        QtCore.QThread.__init__(self, parent)
        self.sound = sound

    def run(self):
        while not self.sound.isFinished():
            self.msleep(1)
