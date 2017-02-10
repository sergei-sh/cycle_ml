
import argparse
import numpy as np
import csv
import tensorflow as tf

from cycle_ml import RecipeData

parser = argparse.ArgumentParser()
parser.add_argument("--b0", type=float)
parser.add_argument("--b1", type=float)
parser.add_argument("--dispersion", type=float)
parser.add_argument("--x_max", type=int)
parser.add_argument("--tool_recipe", type=str)
args = parser.parse_args()

np.random.seed(None)

x_max = int(args.x_max)
counter = 0

if hasattr(args, "tool_recipe"):
    tool_recipe = args.tool_recipe
else:
    tool_recipe = "_data.tf"

x_storage = []
y_storage = []
def gen_points(x_s, y_s):
        y_s = np.multiply(y_s, args.dispersion * 2)
        y_s = np.subtract(y_s, args.dispersion)
        for x, y in zip(x_s, y_s):
            y_line = args.b0 + args.b1 * x
            y = y_line + y
            global x_storage
            global y_storage
            x_storage.append(float(x))
            y_storage.append(float(y))
"""
x_clu20 = np.arange(18, 23)
for _ in range(4):
    y_clu20 = np.random.rand(len(x_clu20))
    gen_points(x_clu20, y_clu20)

x_clu10 = np.arange(8, 13)
for _ in range(4):
    y_clu10 = np.random.rand(len(x_clu10))
    gen_points(x_clu10, y_clu10)
"""

y_main = np.random.rand(x_max)
x_main = np.arange(1, x_max + 1)

gen_points(x_main, y_main)            

RecipeData.save(RecipeData(x_storage, y_storage), tool_recipe)
