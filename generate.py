""" 
Updated: 2017
Author: Sergei Shliakhtin
Contact: xxx.serj@gmail.com
Notes: Generates either csv or tf with points distorted along a given line
"""

import argparse
import numpy as np
import csv
import tensorflow as tf

from cycle_ml import RecipeData

from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("b0", type=float)
parser.add_argument("b1", type=float)
parser.add_argument("dispersion", type=float)
parser.add_argument("x_max", type=int)
parser.add_argument("tool_recipe", type=str)
parser.add_argument("--multiplicator", type=int)
parser.add_argument("csv", action="store_true")
parser.add_argument("--truncate", type=int)
parser.add_argument("--no_clusters", action="store_true")
args = parser.parse_args()

"""Reproducible results are desired"""
np.random.seed(1)

x_max = int(args.x_max)
counter = 0

if hasattr(args, "tool_recipe"):
    tool_recipe = args.tool_recipe
else:
    tool_recipe = "_data.tf"

x_storage = []
y_storage = []
#y_line = []
def gen_points(x_s, y_s):
    """Adds another set of points to x_storage and y_storage
    Args:
        x_s: x-axis values 
        y_s: float (0 .. 1) usually and output of rand(), converted to y-value using dispersion
    """
    y_s = np.multiply(y_s, args.dispersion * 2)
    y_s = np.subtract(y_s, args.dispersion)
    for x, y in zip(x_s, y_s):
        y_line = args.b0 + args.b1 * x
        y = y_line + y
        global x_storage
        global y_storage
        #global y_line
        x_storage.append(float(x))
        y_storage.append(float(y))
        #y_line.append(float(y_line))

mult = args.multiplicator if args.multiplicator else 1
for _ in range(0, mult):
    if args.no_clusters is not None and not args.no_clusters:
        x_clu20 = np.arange(18, 23)
        for _ in range(4):
            y_clu20 = np.random.rand(len(x_clu20))
            gen_points(x_clu20, y_clu20)

        x_clu10 = np.arange(8, 13)
        for _ in range(4):
            y_clu10 = np.random.rand(len(x_clu10))
            gen_points(x_clu10, y_clu10)


    y_main = np.random.rand(x_max)
    x_main = np.arange(1, x_max + 1)

    gen_points(x_main, y_main)            

if args.truncate:
    x_storage[:] = x_storage[:args.truncate]    
    y_storage[:] = y_storage[:args.truncate]    

print(x_storage, y_storage)    

if args.csv:
    class GenPoints:
        """ Csv writer """
        def __init__(self, out_fname):
            self._fname = out_fname

        def __call__(self, x_s, y_s):
            with open(file=self._fname, mode="w", newline="") as out_file:
                writer = csv.writer(out_file)
                for x, y in zip(x_s, y_s):
                    writer.writerow([x, y])
                    print(x, y)
    GenPoints(tool_recipe + ".csv")(x_storage, y_storage)                
else:
    RecipeData.save(RecipeData(x_storage, y_storage, [0 for _ in range(0, len(x_storage))]), tool_recipe)

scaler = StandardScaler()
scaler.fit(y_storage)
print("{} points total, sigma: {}".format(len(x_storage), scaler.var_))
