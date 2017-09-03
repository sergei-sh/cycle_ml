""" 
Updated: 2017
Author: Sergei Shliakhtin
Contact: xxx.serj@gmail.com
Notes:
"""

import unittest
import sys

import tensorflow as tf

sys.path.append("../")

#from cycle_ml.recipe_data import RecipeData: works when running from cycle_ml/tests (with sys.path.append("../"))
# doesn't work from cycle_ml running tests/test_recipe_data.py (with or without...)
from cycle_ml import RecipeData


class TestRecipeData(unittest.TestCase):
    def test_save_load(self):
        try:
            saved = RecipeData([1, 2, 9], [10, 20, 30], [6, 7])
            saved.save("test")
            loaded = RecipeData()
            loaded.load("test")
        except tf.errors.NotFoundError:
            assert False, "Did you create 'models' subdir?"
            
        self.assertEqual(saved.wafer_counts, loaded.wafer_counts)
        self.assertEqual(saved.cycle_times, loaded.cycle_times)
        self.assertEqual(saved.abs_err, loaded.abs_err)

    def test_save_load_pending(self):
        saved = RecipeData([1, 2, 3], [10, 20, 30], [7])
        saved.wc_pending = 4
        saved.predicted_pending = 39.5
        saved.save("test")
        loaded = RecipeData()
        loaded.load("test")
        self.assertEqual(saved.wafer_counts, loaded.wafer_counts)
        self.assertEqual(saved.cycle_times, loaded.cycle_times)
        self.assertEqual(saved.abs_err, loaded.abs_err)
        self.assertEqual(saved.wc_pending, loaded.wc_pending)
        self.assertEqual(saved.predicted_pending, loaded.predicted_pending)
        
        loaded.acquire_pending(41)
        self.assertEqual(loaded.wafer_counts, [1, 2, 3, 4])
        self.assertEqual(loaded.cycle_times, [10, 20, 30, 41])
        self.assertEqual(loaded.abs_err, [7, 1.5])

suite = unittest.TestLoader().loadTestsFromTestCase(TestRecipeData)
unittest.TextTestRunner(verbosity=2).run(suite)
