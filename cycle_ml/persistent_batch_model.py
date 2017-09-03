""" 
Updated: 2017
Author: Sergei Shliakhtin
Contact: xxx.serj@gmail.com
Notes:
"""

import sys

from cycle_ml import Model, aux      

class PersistentBatchModel:
    """Model proxy saving its weights and training the model with one batch at a time
    While Model class performs Batch Gradient Descent, this class cuts down the input,
    so that we're learning mini-batch GD, formed from a continuous input;
    after input size reaches BATCH_SIZE the so-far trained weights are saved and input vector
    is cleared, so that we never use more than O(BATCH_SIZE) memory/disk storage"""
    BATCH_SIZE = 30 

    def __init__(self, *, recipe_data, tool_recipe):
        """Args:
            recipe_data: the model is loaded from given tool_recipe id and re-trained with this
                input vector; note, the vector is cleared once reaching the batch size;
            tool_recipe: to model id to form load/save path
        """
        self.model = Model(tool_recipe=tool_recipe, recipe_data=recipe_data)

        if aux.model_exists(tool_recipe):
            if self.model.load():
                if aux.log_verbose:
                    print("Model loaded")
                else:
                    pass
            else:
                sys.exit("Failed to load the model!!!")
        else:
            if aux.log_verbose:
                print("No pre-trained model yet")

        tail_batch = len(recipe_data)
        assert tail_batch <= self.BATCH_SIZE
        if aux.log_verbose:
            print("Tail batch length {}: ".format(tail_batch))
        if tail_batch:
            self.model.update()
        """This way we adapt online process to be mini-batch mode actually"""
        if self.BATCH_SIZE == len(recipe_data):
            if aux.log_verbose:
                print("Saving model, clearing batch")
            self.model.save()
            recipe_data.clear_batch()

    def predict(self, x):
        """
        Args:
            x: a float - single wafer count
        """
        return self.model.predict([x])[0][0]            

    def test_predict(self, x_lst):
        test_pred = self.model.predict(x_lst)
        print("Predict({}): {}".format(x_lst, test_pred))


