
from collections import deque
import numpy as np
import itertools as itools
from sklearn import preprocessing
import tensorflow as tf
import tensorflow.python.debug as tfdbg

class BadArgumentError(ValueError):
    pass

""" 
labels_batch: 1-D tensor of values N each in 0 .. total_labels - 1
total_labels: number of different labels
return value: one-hot tensor of shape (len(labels_batch), total_labels)
"""
def my_sparse_to_onehot(labels_batch, total_labels):    
    sparse_labels = tf.reshape(labels_batch, [-1, 1])
    derived_size = tf.shape(labels_batch)[0]
    indices_row = tf.range(0, derived_size, 1)
    indices = tf.reshape(indices_row, [-1, 1])
    concated = tf.concat(1, [indices, sparse_labels]) 
    return tf.sparse_to_dense(concated, [derived_size, total_labels], 1.0, 0.0)


class Model:
    def __init__(self):
        pass

    def train(self, *, data=None, max_runs, optimizer):
        if None == data: 
            """Another training round using the same model"""
        else:            
            assert not hasattr(self, "data"), "Will not load data twice in the same model"
            self.data = data
        self.data_len = len(data.wafer_counts)

        self.sess = tf.Session()
        #with self.sess:
        try:
            #print("Starting round with {} inputs, b0 = {:.1f}, b1 = {:.1f}".format(data_len, b0_init, b1_init))
            tool_recipe_width = data.tr_cross.width
            self.x = tf.placeholder(tf.float32,  name="x")
            self.y_act = tf.placeholder(tf.float32, name="y_act")
            #self.y = tf.placeholder(tf.float32,  name="y")
            self.tool_recipe_row = tf.placeholder(tf.int32, name="tool_recipe_row")

            """input validity checking"""
            x_shape = tf.shape(self.x)
            y_shape = tf.shape(self.y_act)
            tr_shape = tf.shape(self.tool_recipe_row)
            self.assert_0 = tf.assert_equal(x_shape, y_shape, [x_shape, y_shape]) 
            self.assert_1 = tf.assert_equal(y_shape, tr_shape, [y_shape, tr_shape])

            z = tf.to_float(my_sparse_to_onehot(self.tool_recipe_row, tool_recipe_width))
            b0 = tf.Variable(np.zeros(tool_recipe_width), dtype=tf.float32, trainable=True, name="b0")
            b1 = tf.Variable(np.zeros(tool_recipe_width), dtype=tf.float32, trainable=True, name="b1")

            """building the model"""
            def category_x_vect(tensor, vect):
                v_col = tf.reshape(vect, [-1, 1])
                prod_col = tf.matmul(tensor, v_col) 
                return tf.reshape(prod_col, [1, -1])
            zb0 = category_x_vect(z, b0)
            zb1 = category_x_vect(z, b1)
            self.y = zb0 + zb1 * self.x

            ERROR_MIN = 1e-6
            error = tf.sqrt(tf.clip_by_value((self.y - self.y_act) * (self.y - self.y_act), clip_value_min=ERROR_MIN, clip_value_max=1e+127))
            #train_step = tf.train.GradientDescentOptimizer(learning_rate=precision).minimize(error)
            train_step = optimizer.minimize(error)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            x_std, y_std = self._transform(self.data.wafer_counts, self.data.cycle_times)
            feed_dict = { self.x: x_std, self.tool_recipe_row: self.data.tool_recipe, self.y_act: y_std }

            #grad1 = tf.gradients(error, [b0])[0]
            #grad2 = tf.gradients(error, [b1])[0]
            fetches_arg = { "b0": b0, b1: b1, "y": self.y, error: error, train_step: train_step, zb1: zb1, 
                            "a0": self.assert_0, "a1": self.assert_1}
            #tf.logging.set_verbosity(tf.logging.DEBUG)
            #out = tf.Print(error, [error])
            for i in range(0, max_runs):
                fetches = self.sess.run(fetches_arg, feed_dict)
                #writer.add_summary(fetches[summaries])

                self.b0 = fetches["b0"]
                self.b1 = fetches[b1]
                self.err = fetches[error]
                if i % 10 == 0:
                    print("Step {}".format(i))
                    self.print_results()

            if i == max_runs - 1:
                print("Finished by {} steps".format(max_runs))
            self.print_results()

            self.y_pred = fetches["y"]
        except Exception as e:
            self.sess.close()
            raise e
            
        self.print_results()         
        print(self.data.tr_cross)

    def print_results(self):
        print("b0 {} b1 {}".format(self.b0, self.b1))
        #for x, y in zip(self.data.wafer_counts,

    """Predict y values, given vector of x values and tool_recipe tuple which applies to all x"""
    def predict(self, x_vect, tool_recipe):
        tr_int = self.data.tr_cross.get_int(tool_recipe)
        tr_vect = list(itools.repeat(tr_int, len(x_vect)))
        """y_act input is dummy for this case"""
        y_act = list(itools.repeat(0, len(x_vect)))
        x_vect, y_act = self._transform(x_vect, y_act)
        feed_dict = { self.x: x_vect, self.tool_recipe_row: tr_vect, self.y_act: y_act}
        y, _, _ = self.sess.run(fetches=[self.y, self.assert_0, self.assert_1], feed_dict=feed_dict)
        return self._inverse_y(y)[0] 

    def _transform(self, x_in, y_in):
        if not hasattr(self, "x_scaler"):
            self.x_scaler = preprocessing.StandardScaler().fit(x_in)
            self.y_scaler = preprocessing.StandardScaler().fit(y_in)
        x_std = self.x_scaler.transform(x_in)
        y_std = self.y_scaler.transform(y_in)
        return x_std, y_std

    def _inverse_y(self, y_std):
       return self.y_scaler.inverse_transform(y_std) 



