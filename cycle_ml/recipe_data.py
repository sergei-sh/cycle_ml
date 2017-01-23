
import numpy as np

class RecipeData:
    def __init__(self):
       self.wafer_counts = np.array([], dtype=int) 
       self.cycle_times = np.array([])
       self.tool_recipe = []
       self.tr_cross = None

    def __str__(self):
        msg = ""
        for i in range(0, len(self.wafer_counts)):
            msg += "{} {} {}; \n ".format(self.tool_recipe[i], self.wafer_counts[i], self.cycle_times[i])
        return msg


