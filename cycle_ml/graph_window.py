
from PyQt4 import QtGui, QtCore

import pyqtgraph as pg
import numpy as np

class GraphWindow(pg.PlotWidget):
    def __init__(self, parent, showable):
        super().__init__(parent)
        self.showable = showable()
        self.show_next()
        self.closing = False

    def show_next(self):
        try:
            self.show_model(next(self.showable))
        except StopIteration:
            self.closing = True
            self.close()

    def show_model(self, data):
        curve = self.plot(pen='g')
        
        #x1 = model.data.wafer_counts[-1:]
        #curve.setData(
        #    np.array([0, x1]), 
        #    np.array([model.b0, model.b0 + model.b1 * x1]))

        s1 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush("b"))
        s1.addPoints(data.train.x, data.train.y)
        self.addItem(s1)

        s0 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush("r"))
        s0.addPoints(data.test.x, data.test.y)
        self.addItem(s0)

    def keyPressEvent(self, event):
        self.show_next()
        event.accept()

    def focusOutEvent(self, event):
        if not self.closing:
            self.close()
        event.accept()

       

        
