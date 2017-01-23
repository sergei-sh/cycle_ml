
import sys

from collections import namedtuple as ntuple
import pyqtgraph as pg
from PyQt4 import QtGui
import tensorflow as tf

from cycle_ml import Loader
from cycle_ml import Model
from cycle_ml import preprocessor
from collections import deque

from cycle_ml.application import run

from cycle_ml.graph_window import GraphWindow

def showable():        
    model = Model()
    data = Loader.get_data(sys.argv[1])
    data = preprocessor.get_data(data)
    model.train(data=data, max_runs=800, optimizer=tf.train.AdamOptimizer(learning_rate=0.01))

    MyDataSet = ntuple("MyDataSet", ["x", "y"])
    train = MyDataSet([], [])
    tool_recipe = ('endura', 'rec2-10')
    tr_int=data.tr_cross.get_int(tool_recipe)
    for x, y, tr in zip(data.wafer_counts, data.cycle_times, data.tool_recipe):
        if tr_int == tr:
            train.x.append(x)
            train.y.append(y)
    test = MyDataSet([], [])
    test.x.extend(range(0, 30))
    test.y.extend(model.predict(test.x, tool_recipe))
    print(test.y[0], test.y[1])
    graph_data = ntuple("train", "test")
    graph_data.train = train
    graph_data.test = test
    yield graph_data 

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


