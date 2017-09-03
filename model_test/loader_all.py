"""Takes an input csv in the following format:
WAFER_COUNT, CYCLE_TIME
""" 
import csv
import sys

class Data:
    def __init__(self, x, y):
        self.wafer_counts = x
        self.cycle_times = y 

def load_all(fname):
    print("Reading {}".format(fname))
    x = []
    y = []
    with open(fname, newline="") as csv_file:
        reader = csv.reader(csv_file)
        step = 0
        for row in reader:
            assert len(row) == 2, "Should have 2 columns"
            wc = float(row[0])
            ct = float(row[1])
            x.append(wc) 
            y.append(ct)

    return Data(x, y)            
                

             
