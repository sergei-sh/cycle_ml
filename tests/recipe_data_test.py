
import unittest
import sys

sys.path.append("../")

#from cycle_ml.recipe_data import RecipeData: works when running from cycle_ml/tests (with sys.path.append("../"))
# doesn't work from cycle_ml running tests/test_recipe_data.py (with or without...)
from cycle_ml import RecipeData


class TestRecipeData(unittest.TestCase):
    def test_save_load(self):
        saved = RecipeData([1, 2, 9], [10, 20, 30], [12, 21, 31])
        saved.save("test")
        loaded = RecipeData()
        loaded.load("test")
        self.assertEqual(saved.wafer_counts, loaded.wafer_counts)
        self.assertEqual(saved.cycle_times, loaded.cycle_times)
        self.assertEqual(saved.predicted, loaded.predicted)

    def test_save_load_pending(self):
        saved = RecipeData([1, 2, 3], [10, 20, 30], [12, 21, 31])
        saved.wc_pending = 4
        saved.predicted_pending = 39.5
        saved.save("test")
        loaded = RecipeData()
        loaded.load("test")
        self.assertEqual(saved.wafer_counts, loaded.wafer_counts)
        self.assertEqual(saved.cycle_times, loaded.cycle_times)
        self.assertEqual(saved.predicted, loaded.predicted)
        self.assertEqual(saved.wc_pending, loaded.wc_pending)
        self.assertEqual(saved.predicted_pending, loaded.predicted_pending)
        
        loaded.acquire_pending(41)
        self.assertEqual(loaded.wafer_counts, [1, 2, 3, 4])
        self.assertEqual(loaded.cycle_times, [10, 20, 30, 41])
        self.assertEqual(loaded.predicted, [12, 21, 31, 39.5])
         
suite = unittest.TestLoader().loadTestsFromTestCase(TestRecipeData)
unittest.TextTestRunner(verbosity=2).run(suite)
