from PyQt5 import QtGui
from PyQt5.QtCore import QPoint, pyqtSignal
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import QLabel, QWidget


class MyLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)

        self.piano_boundary = []

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if len(self.piano_boundary) < 4:
            self.piano_boundary.append(QPoint(ev.x(), ev.y()))
        self.clicked.emit()

    def paintEvent(self, e):
        super(MyLabel, self).paintEvent(e)
        qp = QPainter()
        qp.begin(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setPen(QColor(0, 255, 0))
        qp.drawPolygon(self.piano_boundary)
        qp.end()
