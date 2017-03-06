""" The main project functionality - linear regression model resides here
"""

from collections import deque
import numpy as np
import itertools as itools
from sklearn import preprocessing
import tensorflow as tf
import tensorflow.python.debug as tfdbg

from cycle_ml.aux import get_model_path
from cycle_ml import aux

class BadArgumentError(ValueError):
    pass

class Model:
    """Linear regression model.
    Operates on a specific tool_recipe identifier.
    Uses whole recipe_data to perform Batch Gradient Descent update of the 2 parameters. 
    Weights are initialized to 0 zero and then saved/loaded by calling corresponding functions by a class user.
    The mean and variance used for normalization are calculated on the first batch; Then they are saved into TF model
    along with weights; On consequent load()'s scaling parameters always remain the same as for the first batch.
    """
    LEARNING_RATE = 0.0001 
    ITERATIONS = 3000

    def __init__(self, *, tool_recipe, recipe_data):
        """
        Initializes the model and gets the dataset to learn from. The actual learning happens in update(),
        but the caller might be willing to call load() in between.

        Args:
            tool_recipe: the model id; each tool_recipe has its own weights/scales saved;
            recipe_data: the dataset from which to updated the weights
        """
        self.data = recipe_data
        self.sess = tf.Session()
        self.model_path = get_model_path(tool_recipe)
        try:
            self.x = tf.placeholder(tf.float32,  name="x")
            self.y_act = tf.placeholder(tf.float32, name="y_act")

            """input validity checking"""
            x_shape = tf.shape(self.x)
            y_act_shape = tf.shape(self.y_act)
            self.assert_0 = tf.assert_equal(x_shape, y_act_shape, [x_shape, y_act_shape]) 

            self.b0 = tf.Variable([0], dtype=tf.float32, trainable=True, name="b0")
            self.b1 = tf.Variable([0], dtype=tf.float32, trainable=True, name="b1")

            """building the model"""
            self.y = self.b0 + self.b1 * self.x
            self.error = abs(self.y - self.y_act)
            """x.scale_, x.mean_, y.scale_, y.mean_. will be replaced on session load, if any. They are re-calculated
            on each __init__ just to simplify the code"""
            self.x_scaler = preprocessing.StandardScaler().fit(_sample_feature_matrix(self.data.wafer_counts)) 
            self.y_scaler = preprocessing.StandardScaler().fit(_sample_feature_matrix(self.data.cycle_times))
            scalers = [self.x_scaler.scale_[0], self.x_scaler.mean_[0], self.y_scaler.scale_[0], self.y_scaler.mean_[0]]
            self.scalers = tf.Variable(scalers, dtype=tf.float32, name="scalers", validate_shape=True)

            optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.LEARNING_RATE)
            self.train_step = optimizer.minimize(self.error)
            self.sess.run(tf.global_variables_initializer())

        except Exception as e:
            self.sess.close()
            raise e

    def update(self):
        """The actual learning takes place here; the dataset is set in __init__() and load() is 
        called if needed
        """
        try:
            x_std, y_std = self._transform(self.data.wafer_counts, self.data.cycle_times)
            feed_dict = { self.x: x_std, self.y_act: y_std }
            fetches_arg = {  "b0": self.b0, "b1": self.b1, "y": self.y, "error": self.error, 
                "train_step": self.train_step, "a0": self.assert_0}#, "init_scalers": self.init_scalers}

            if aux.log_verbose:
                print("Starting from: ", self.sess.run([self.b0, self.b1, self.scalers]))
            for i in range(0, self.ITERATIONS):
                fetches = self.sess.run(fetches_arg, feed_dict)

                self.fetches = fetches

            if aux.log_verbose:
                self.print_results()
        except Exception as e:
            self.sess.close()
            raise e

    def load(self):
        """Loads the model.
        The load path is calculated in advance from tool_recipe in __init__()
        """
        try:
            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_path) 

            scalers = self.sess.run([self.scalers])[0]
            self.x_scaler = preprocessing.StandardScaler()
            self.x_scaler.scale_ = scalers[0] 
            self.x_scaler.mean_ = scalers[1] 
            self.y_scaler = preprocessing.StandardScaler()
            self.y_scaler.scale_ = scalers[2] 
            self.y_scaler.mean_ = scalers[3] 
            return True
        except tf.errors.NotFoundError:
            return False
            
    def save(self):
        """Saves model weights and normalization parameters"""
        var_list = [self.b0, self.b1, self.scalers]
        if aux.log_verbose:
           print("saving", self.sess.run(var_list))
        saver = tf.train.Saver(var_list)
        saver.save(self.sess, self.model_path)

    def print_results(self):
        """ """
        print("b0 {} b1 {} ".format(self.fetches["b0"], self.fetches["b1"]))

    def predict(self, x_vect):
        """Predict y values, given vector of x values and tool_recipe tuple which applies to all x"""
        y_act = list(itools.repeat(0, len(x_vect)))
        x_vect, y_act = self._transform(x_vect, y_act)
        feed_dict = { self.x: x_vect, self.y_act: y_act}
        y, _, = self.sess.run(fetches=[self.y, self.assert_0], feed_dict=feed_dict)
        return self._inverse_y(y) 

    def _transform(self, x_in, y_in):
        """Feature normalization"""
        x_std = self.x_scaler.transform(_sample_feature_matrix(x_in))
        y_std = self.y_scaler.transform(_sample_feature_matrix(y_in))
        return x_std, y_std

    def _inverse_y(self, y_std):
       """Un-normalize y-vector according to saved scale parameters"""
       return self.y_scaler.inverse_transform(_sample_feature_matrix(y_std)) 

def _sample_feature_matrix(in_list):
    """Converts a list of samples with a single feature value in each to (n_samples, n_features) matrix 
        suitable for sklearn"""
    return np.array(in_list).reshape(-1, 1).astype(np.float32)



