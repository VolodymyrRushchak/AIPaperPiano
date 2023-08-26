from PyQt5 import QtGui
from PyQt5.QtCore import QPoint, pyqtSignal
from PyQt5.QtWidgets import QLabel, QWidget


class ClickableLabel(QLabel):
    clicked = pyqtSignal(QtGui.QMouseEvent)

    def __init__(self, parent: QWidget):
        super().__init__(parent)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        self.clicked.emit(ev)

