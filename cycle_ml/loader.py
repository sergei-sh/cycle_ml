
import csv
import os
import numpy as np
import tensorflow as tf
from os import walk

from cycle_ml.recipe_data import RecipeData

class Loader:
    def __init__(self, load_path):
        try:
            
            print("Looking at path {}".format(load_path))
            self.data = RecipeData()
            for (dirpath, dirnames, filenames) in walk(load_path):
                filenames = list(filter(lambda s: s.endswith(".csv"), filenames))
                fname = os.path.join(dirpath, filenames[0])
                print("Reading {}".format(fname))
                with open(fname, newline="") as csv_file:
                    reader = csv.reader(csv_file)
                    for row in reader:
                        assert len(row) == 2, "Should have 2 columns"
                        self.data.wafer_counts = np.append(self.data.wafer_counts, [int(row[0])])
                        self.data.cycle_times = np.append(self.data.cycle_times, [float(row[1])])
                print("{} rows loaded".format(len(self.data.wafer_counts)))
            
        except Exception as e:
           raise e
           

             
