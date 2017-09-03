""" 
Updated: 2017
Author: Sergei Shliakhtin
Contact: xxx.serj@gmail.com
Notes:

Load/display dataset points in Qt window
"""
from PyQt4 import QtGui

import pyqtgraph as pg

from loader_all import load_all

app = QtGui.QApplication([])
wnd = pg.PlotWidget()

wnd.getPlotItem().setLabel(axis="left", text="Cycle time")
wnd.getPlotItem().setLabel(axis="bottom", text="Wafer count")
#wnd.getPlotItem().setTitle("C=1000, kernel='rbf', gamma=0.0105")
s1 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush("b"))

d_data = load_all("demo.csv")
s1.addPoints(d_data.wafer_counts, d_data.cycle_times)
wnd.addItem(s1)

wnd.show()
app.exec_()


