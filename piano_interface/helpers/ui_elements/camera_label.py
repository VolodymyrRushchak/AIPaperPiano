from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import QWidget

from helpers.ui_elements.clickable_label import ClickableLabel


class CameraLabel(ClickableLabel):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.piano_boundary = []

    def paintEvent(self, e):
        super().paintEvent(e)
        qp = QPainter()
        qp.begin(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.setPen(QColor(0, 255, 0))
        qp.drawPolygon(self.piano_boundary)
        qp.end()
