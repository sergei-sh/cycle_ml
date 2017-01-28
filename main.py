
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

from cycle_ml.recipe_data import RecipeData
class DataPart(object):
    def __init__(self, recipe_data, length):
        self.wafer_counts = recipe_data.wafer_counts[:length]
        self.cycle_times = recipe_data.cycle_times[:length]

def showable():        
    loader = Loader(sys.argv[1])

    data = loader.data
    #start_idx = len(data.wafer_counts) - 20
    start_idx = 1
    for length in range(start_idx, len(data.wafer_counts) - 1):
        model = Model()
        data_p = DataPart(data, length)
        print(data_p.wafer_counts)
        model.train(data=data_p, max_runs=200, optimizer=tf.train.AdamOptimizer(learning_rate=0.01))
        pred = data.wafer_counts[length + 1:length+2]
        pred_x = pred[0]
        y = model.predict(pred)[0]
        check = 10 + 5 * pred_x 
        del model
        print("Predict {}: {}, check: {} abs.err.:{}".format(pred_x, y, check, abs(y - check)))
        del pred_x

    MyDataSet = ntuple("MyDataSet", ["x", "y"])
    train = MyDataSet([], [])
    train.x.extend(data.wafer_counts)
    train.y.extend(data.cycle_times)
    test = MyDataSet([], [])
    test.x.extend(range(0, 30))
    test.y.extend(model.predict(test.x))
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


