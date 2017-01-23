
import csv
import os
import numpy as np
from os import walk

from cycle_ml.recipe_data import RecipeData

class Loader:
    def __init__(self):
        pass

    def get_data(load_path):
        try:
            print("Looking at path {}".format(load_path))
            data = RecipeData()
            for (dirpath, dirnames, filenames) in walk(load_path):
                filenames = list(filter(lambda s: s.endswith(".csv"), filenames))
                for fname in filenames:
                    tool, recipe_ext = fname.split("_")
                    recipe = recipe_ext.split(".")[0]
                    assert tool and recipe
                    fpathname = os.path.join(dirpath, fname)
                    print("Reading {}, tool: {}, recipe: {}".format(fpathname, tool, recipe))
                    with open(fpathname, newline="") as csv_file:
                        reader = csv.reader(csv_file)
                        for row in reader:
                            assert len(row) == 2, "Should have 2 columns"
                            data.wafer_counts = np.append(data.wafer_counts, [int(row[0])])
                            data.cycle_times = np.append(data.cycle_times, [float(row[1])])
                            data.tool_recipe.append((tool,recipe))
                    print("Total {} rows loaded".format(len(data.wafer_counts)))
            return data                    
                    
        except Exception as e:
            raise e


         
