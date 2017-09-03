""" 
Updated: 2017
Author: Sergei Shliakhtin
Contact: xxx.serj@gmail.com
Notes:

Takes an input csv in the following format:
TOOL, WAFER_COUNT, RECIPE, CYCLE_TIME, SEQUENCE_NO
and sends it to main.py according to usual protocol (see README)
""" 

import argparse
import csv
import sys

from cycle_ml.aux import my_call

parser = argparse.ArgumentParser()
parser.add_argument("fname", default="", type=str)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

fname = args.fname
print("Reading {}".format(fname))
with open(fname, newline="") as csv_file:
    reader = csv.reader(csv_file)
    """Skip header"""
    for row in reader:
        break
    for row in reader:
        try:
            assert len(row) == 5, "Should have 5 columns"
            tool = row[0]
            wc = float(row[1])
            recipe = row[2]
            ct = float(row[3])
            sequence = int(row[4])
            tool_recipe = tool + "," + recipe
            print("\n")
            cmd = ["python", "main.py", "--next_datapoint", str(wc), tool_recipe]
            if args.verbose:
                cmd.append("--verbose")
            if 0 != my_call(cmd):
                sys.exit(1)
            print("\n")
            my_call(["python", "main.py", "--finish_datapoint", str(ct), tool_recipe, "--sequence", str(sequence)])

        except KeyboardInterrupt:
            sys.exit(0)
            

         
