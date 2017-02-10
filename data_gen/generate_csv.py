
import argparse
import numpy as np
import csv

parser = argparse.ArgumentParser()
parser.add_argument("b0", type=float)
parser.add_argument("b1", type=float)
parser.add_argument("dispersion", type=float)
parser.add_argument("x_max", type=int)
parser.add_argument("out_file", type=str)
parser.add_argument("--with_clusters", action="store_true")
args = parser.parse_args()

np.random.seed(None)

x_max = int(args.x_max)
counter = 0

out_fname = args.out_file
print("Writing to {}".format(out_fname))
open(file=args.out_file, mode="w").close()

class GenPoints:
    def __init__(self, out_fname):
        self._fname = out_fname

    def __call__(self, x_s, y_s):
        with open(file=self._fname, mode="a", newline="") as out_file:
            writer = csv.writer(out_file)
            y_s = np.multiply(y_s, args.dispersion * 2)
            y_s = np.subtract(y_s, args.dispersion)
            for x, y in zip(x_s, y_s):
                y_line = args.b0 + args.b1 * x
                y = y_line + y
                writer.writerow([x, y])
                global counter
                counter += 1

gen_points = GenPoints(out_fname)                

y_main = np.random.rand(x_max)
x_main = np.arange(1, x_max + 1)

gen_points(x_main, y_main)            

if (args.with_clusters):
    x_clu20 = np.arange(18, 23)
    for _ in range(4):
        y_clu20 = np.random.rand(len(x_clu20))
        gen_points(x_clu20, y_clu20)

    x_clu10 = np.arange(8, 13)
    for _ in range(4):
        y_clu10 = np.random.rand(len(x_clu10))
        gen_points(x_clu10, y_clu10)


print("{} points generated".format(counter))

