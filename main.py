
import sys

import argparse
from collections import namedtuple as ntuple
import tensorflow as tf

from cycle_ml import Model
from collections import deque

from cycle_ml.recipe_data import RecipeData, mae10

def showable(model, rdata):        
    print("run")
    MyDataSet = ntuple("MyDataSet", ["x", "y"])
    train = MyDataSet([], [])
    train.x.extend(rdata.wafer_counts)
    train.y.extend(rdata.cycle_times)
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
        parser = argparse.ArgumentParser()
        #parser.add_argument("--graph", action="store_true")
        parser.add_argument("--next_datapoint", type=int)
        parser.add_argument("--finish_datapoint", type=float)
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("tool_recipe", nargs="?", default="", type=str)
        args = parser.parse_args()
        args.graph = 0

        if not args.tool_recipe:
            parser.print_help()
            sys.exit("\n Need to specify tool,recipe")

        elif None != args.next_datapoint or args.graph:
            print("Loading the model...")
            rdata = RecipeData()
            rdata.load(args.tool_recipe) 
            more = 2 - len(rdata)
            rdata.wc_pending = args.next_datapoint
            if more <= 0:
                model = Model()
                model.update(rdata)
                if args.graph:
                    from cycle_ml.application import run
                    run(showable(model, rdata))
                    exit(0)
                assert args.next_datapoint > 0
                pred = model.predict([args.next_datapoint])[0][0]
                print("Predicted cycle time (s): {:.4f}".format(pred))
                if args.verbose:
                    test_pred = model.predict([0, 1])
                    print("Predict(0): {}, Predict(1): {}".format(test_pred[0], test_pred[1]))
                print("MAE(last 10)(s): {:.4f}".format(mae10(rdata)))
                rdata.predicted_pending = pred
            else:
                print("{} more finished datapoints to start predictions".format(more))
            rdata.save(args.tool_recipe)

        elif args.finish_datapoint:
            rdata = RecipeData()
            rdata.load(args.tool_recipe)
            if rdata.wc_pending:
                print("Acquired cycle time {} for wafer count {}".format(args.finish_datapoint, rdata.wc_pending))
                rdata.acquire_pending(args.finish_datapoint)
                rdata.save(args.tool_recipe)
                if args.verbose:
                    print(rdata)
            else:
                sys.exit("No datapoint started currently for {}".format(args.tool_recipe))
             

