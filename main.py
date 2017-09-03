""" 
Updated: 2017
Author: Sergei Shliakhtin
Contact: xxx.serj@gmail.com
Notes:

The program entry point. See README for usage"""


import argparse
from collections import namedtuple as ntuple
from collections import deque
import sys

import tensorflow as tf

from cycle_ml import PersistentBatchModel

from cycle_ml.recipe_data import RecipeData, mae10
from cycle_ml import aux
from cycle_ml.aux import log_tool

def showable(model, rdata):        
    """To be removed"""
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
        parser = argparse.ArgumentParser()
        #parser.add_argument("--graph", action="store_true")
        parser.add_argument("--next_datapoint", type=float)
        parser.add_argument("--finish_datapoint", type=float)
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("tool_recipe", nargs="?", default="", type=str)
        parser.add_argument("--sequence", default=0, type=int)
        args = parser.parse_args()
        args.graph = 0

        aux.log_verbose = args.verbose

        if not args.tool_recipe:
            parser.print_help()
            sys.exit("\n Need to specify tool,recipe")

        elif None != args.next_datapoint or args.graph:
            print("Loading the model...")
            rdata = RecipeData()
            rdata.load(args.tool_recipe) 
            MIN_POINTS = 1 
            more = MIN_POINTS - len(rdata)
            rdata.wc_pending = args.next_datapoint
            if more <= 0:
                model = PersistentBatchModel(recipe_data=rdata, tool_recipe=args.tool_recipe)
                assert args.next_datapoint > 0
                pred = model.predict(args.next_datapoint)
                print("Predicted cycle time (s): {:.4f}".format(pred))
                mae = mae10(rdata) 
                if mae:
                    print("MAE(last 10)(s): {:.4f}".format(mae))
                rdata.predicted_pending = pred
            else:
                print("{} more finished datapoints to start predictions".format(more))
            rdata.save(args.tool_recipe)

        elif args.finish_datapoint:
            rdata = RecipeData()
            rdata.load(args.tool_recipe)

            """For logging only"""
            predicted = rdata.predicted_pending
            mae10_val = mae10(rdata)
            wafer_count = rdata.wc_pending

            if rdata.wc_pending:
                print("Acquired cycle time {} for wafer count {}".format(args.finish_datapoint, rdata.wc_pending))
                abs_err = rdata.acquire_pending(args.finish_datapoint)
                rdata.save(args.tool_recipe)
                print("Absolute error: ", abs_err)
                first = (not aux.model_exists(args.tool_recipe)) and  1 == len(rdata)
                log_tool(tool_recipe=args.tool_recipe, the_first=first, wafer_count=wafer_count, 
                    predicted=predicted, actual=args.finish_datapoint, abs_err=abs_err, mae10=mae10_val, sequence=args.sequence)    
                sys.exit(int(abs_err))
            else:
                sys.exit("No datapoint started currently for {}".format(args.tool_recipe))
             

