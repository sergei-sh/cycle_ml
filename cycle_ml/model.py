
from collections import deque
import numpy as np
import itertools as itools
from sklearn import preprocessing
import tensorflow as tf
import tensorflow.python.debug as tfdbg

from cycle_ml.aux import get_model_path

class BadArgumentError(ValueError):
    pass

class Model:
    def __init__(self):
        #self.model_path = get_model_path(tool_recipe)
        self.sess = tf.Session()
        try:
            self.x = tf.placeholder(tf.float32,  name="x")
            self.y_act = tf.placeholder(tf.float32, name="y_act")

            """input validity checking"""
            x_shape = tf.shape(self.x)
            y_act_shape = tf.shape(self.y_act)
            self.assert_0 = tf.assert_equal(x_shape, y_act_shape, [x_shape, y_act_shape]) 

            b0 = tf.Variable([0], dtype=tf.float32, trainable=True, name="b0")
            b1 = tf.Variable([0], dtype=tf.float32, trainable=True, name="b1")

            """building the model"""
            self.y = b0 + b1 * self.x
        except Exception as e:
            self.sess.close()
            raise e

    def update(self, rdata):
        self._update(data=rdata, max_runs=200, optimizer=tf.train.AdamOptimizer(learning_rate=0.01))
 
    def _update(self, *, data=None, max_runs, optimizer):
        try:
            error = abs(self.y - self.y_act)
            train_step = optimizer.minimize(error)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            x_std, y_std = self._transform(data.wafer_counts, data.cycle_times)
            feed_dict = { self.x: x_std, self.y_act: y_std }
            #"b0": b0, "b1": b1,
            fetches_arg = {  "y": self.y, "error": error, "train_step": train_step, "a0": self.assert_0 }

            for i in range(0, max_runs):
                fetches = self.sess.run(fetches_arg, feed_dict)

                self.fetches = fetches
                if i % 10 == 0:
                    self.print_results()

            self.print_results()
        except Exception as e:
            self.sess.close()
            raise e
            
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.model_path)

    def print_results(self):
        pass

    def load(self):
        try:
            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_path) 
            return True
        except tf.errors.NotFoundError:
            return False

    """Predict y values, given vector of x values and tool_recipe tuple which applies to all x"""
    def predict(self, x_vect):
        """y_act input is dummy for this case"""
        y_act = list(itools.repeat(0, len(x_vect)))
        x_vect, y_act = self._transform(x_vect, y_act)
        feed_dict = { self.x: x_vect, self.y_act: y_act}
        y, _, = self.sess.run(fetches=[self.y, self.assert_0], feed_dict=feed_dict)
        return self._inverse_y(y) 

    def _transform(self, x_in, y_in):
        if not hasattr(self, "x_scaler"):
            self.x_scaler = preprocessing.StandardScaler().fit(_sample_feature_matrix(x_in)) 
            self.y_scaler = preprocessing.StandardScaler().fit(_sample_feature_matrix(y_in))
        x_std = self.x_scaler.transform(_sample_feature_matrix(x_in))
        y_std = self.y_scaler.transform(_sample_feature_matrix(y_in))
        return x_std, y_std

    def _inverse_y(self, y_std):
       return self.y_scaler.inverse_transform(_sample_feature_matrix(y_std)) 

"""Converts a list of samples with a single feature value in each to (n_samples, n_features) matrix 
    suitable for sklearn"""
def _sample_feature_matrix(in_list):
    return np.array(in_list).reshape(-1, 1).astype(np.float32)



