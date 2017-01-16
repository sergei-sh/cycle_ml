
import sys

from PyQt4 import QtGui
import tensorflow as tf
import pyqtgraph as pg

from cycle_ml import Loader
from cycle_ml import Model
from collections import deque

from cycle_ml.application import run

from cycle_ml.graph_window import GraphWindow

def showable():        
    model = Model()
    loader = Loader(sys.argv[1])
    model.train(data=loader.data, max_runs=len(loader.data.wafer_counts)*3, optimizer=tf.train.AdamOptimizer(learning_rate=0.1))
    yield model

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: python main.py in_data.csv")
    else:
        need_graph = (len(sys.argv) == 3 and sys.argv[2] == "-graph")
        if need_graph:
            run(showable)
        else:
            it = showable()
            deque(it) 


