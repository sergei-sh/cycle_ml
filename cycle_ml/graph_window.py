
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

    def show_model(self, model):
        curve = self.plot(pen='g')
        x1 = model.data.wafer_counts[-1:]
        curve.setData(
            np.array([0, x1]), 
            np.array([model.b0, model.b0 + model.b1 * x1]))

        s1 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush("b"))
        s1.addPoints(model.data.wafer_counts, model.data.cycle_times)
        self.addItem(s1)

    def keyPressEvent(self, event):
        self.show_next()
        event.accept()

    def focusOutEvent(self, event):
        if not self.closing:
            self.close()
        event.accept()

       

        
