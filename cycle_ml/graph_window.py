
from PyQt4 import QtGui, QtCore

import sys
import pyqtgraph as pg
import numpy as np

class GraphWindow(pg.PlotWidget):
    def __init__(self, parent, showable):
        super().__init__(parent)
        self.showable = showable
        self.show_next()
        self.closing = False

    def show_next(self):
        try:
            self.show_model(next(self.showable))
        except StopIteration:
            sys.exit(0)
            self.closing = True
            self.close()

    def show_model(self, data):
        curve = self.plot(pen='g')
        
        s1 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush("b"))
        s1.addPoints(data.train.x, data.train.y)
        self.addItem(s1)

        s0 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush("r"))
        s0.addPoints(data.test.x, data.test.y)
        self.addItem(s0)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Enter or event.key() == QtCore.Qt.Key_Space:
            self.show_next()
        event.accept()

    def focusOutEvent(self, event):
        if not self.closing:
            self.close()
        event.accept()

       

        
