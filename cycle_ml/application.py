""" 
Updated: 2017
Author: Sergei Shliakhtin
Contact: xxx.serj@gmail.com
Notes: Qt application runner
"""

from PyQt4 import QtGui
from .graph_window import GraphWindow

import pyqtgraph as pg

def run(showable):
    app = QtGui.QApplication([])
    wnd = GraphWindow(None, showable)
    wnd.show()
    app.exec_()

