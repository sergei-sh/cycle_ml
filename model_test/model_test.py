""" 
Updated: 2017
Author: Sergei Shliakhtin
Contact: xxx.serj@gmail.com
Notes:

Research code. Produce various stats/graphs on the model developed.
"""

from collections import deque
import random
import numpy as np
import itertools as itools
import tensorflow as tf
import tensorflow.python.debug as tfdbg
from sklearn import preprocessing

class BadArgumentError(ValueError):
    pass

class Model:
    def __init__(self):
            pass

    def train(self, *, data=None, iterations, batch_size, repeats, precision):
        if None == data: 
            """Another training round using the same model"""
            b0_init = self.b0
            b1_init = self.b1
            data_len = len(self.data.wafer_counts)
        else:            
            assert not hasattr(self, "data"), "Will not load data twice in the same model"
            data_len = len(data.wafer_counts)
            assert iter(data.wafer_counts) is not iter(data.wafer_counts), "Need container argument"
            if data_len < 2: raise BadArgumentError("Need 2 or more samples")
            next_i = len(data.wafer_counts) - 1
            if data.wafer_counts[next_i] == data.wafer_counts[0]: raise BadArgumentError("Need different wafer counts (x)")
            """Chooose MAX"""
            b1_init = (data.cycle_times[next_i] - data.cycle_times[0]) / (data.wafer_counts[next_i] - data.wafer_counts[0])
            b0_init = data.cycle_times[0] - b1_init * data.wafer_counts[0]
            self.data = data
            b0_init = 0 
            b1_init = 0 

        with tf.Session() as sess:
            print("Starting round with {} inputs, b0 = {:.1f}, b1 = {:.1f}".format(data_len, b0_init, b1_init))
            self.x = tf.placeholder(tf.float32, name="x")
            #y = tf.placeholder(tf.float32, name="y")
            b0 = tf.Variable([b0_init], dtype=tf.float32, trainable=True, name="b0")
            b1 = tf.Variable([b1_init], dtype=tf.float32, trainable=True, name="b1")
            out = tf.Print(b0, [b0])
            #the model
            self.y = tf.add(tf.mul(self.x, b1), b0)
            self.y_act = tf.placeholder(tf.float32,  name="y_act")
            ERROR_MIN = 1e-127
               #error = tf.sqrt(tf.clip_by_value((self.y - self.y_act) * (self.y - self.y_act), clip_value_min=ERROR_MIN, clip_value_max=1e+127))
            #error = tf.clip_by_value((self.y - self.y_act) * (self.y - self.y_act), clip_value_min=ERROR_MIN, clip_value_max=1e+127)
            error = tf.squared_difference(self.y, self.y_act)
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate=precision).minimize(error)
            #self.train_step = optimizer.minimize(error)
            #x_in = self.data.wafer_counts
            #y_in = self.data.cycle_times
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            total_y = tf.placeholder(tf.float32)
            total_y_pred = tf.placeholder(tf.float32)
            writer = tf.summary.FileWriter("log_graph", sess.graph)
            total_error = tf.squared_difference(total_y, total_y_pred)
            mse = tf.reduce_mean(total_error)
            tf.summary.scalar("error", mse)
            summaries = tf.summary.merge_all()
            #out = tf.Print(mse, [mse])
             
            grad1 = tf.gradients(error, [b0])[0]
            grad2 = tf.gradients(error, [b1])[0]
            fetches_in = { b0: b0, b1: b1, self.y: self.y, error: error, self.train_step: self.train_step, grad1: grad1, grad2: grad2}


            """Results:
            Non-standardized:
                SGD: diverges
                Adam: converges
                Split:
                    Adam: converges 
            Standardized:
                Adam: converges
                Ftrl: diverges
                Split:
                    Adam: diverges
                    Ftrl: diverges
            """

            wafer_counts, cycle_times = self._transform(self.data.wafer_counts, self.data.cycle_times)

            print(self._transform([0], [640]))
            print(self.y_scaler.get_params())

            batch = 0
            it = 0
            size = len(wafer_counts)
            print(wafer_counts)
            if 1 == batch_size:
                iterable = zip(itools.cycle(wafer_counts), itools.cycle(cycle_times))
            else:
                iterable = zip(
                        [wafer_counts[rng:rng+batch_size] \
                         for rng \
                         in range(0, size, batch_size)],
                        [cycle_times[rng:rng+batch_size] \
                         for rng \
                         in range(0, size, batch_size)]);
            for x_part, y_act_part in  iterable:
                fetches = None
                #print("Batch {} (size: {})".format(batch, len(x_part)))
                #print(self.predict(sess, [50]))
                batch += 1
                for i in range(0, repeats):
                    #print("Feeding {} {}".format([x_part], [y_act_part]))
                    feed_dict = { self.x: [x_part], self.y_act: [y_act_part] }
                    fetches = sess.run(fetches_in, feed_dict)

                    self.b0 = fetches[b0][0]
                    self.b1 = fetches[b1][0]

                    if 0 == it % 20:
 
                        #error over all training set
                        y_pred = self.predict(sess, data.wafer_counts)
                        mse = tf.reduce_mean(tf.squared_difference(data.cycle_times, y_pred))
                        mse_out = sess.run(mse, {})
                        print(mse_out, it)
                      
                        [summ] = sess.run([summaries], {total_y : data.cycle_times, total_y_pred : y_pred})
                        writer.add_summary(summ, it)

                    it += 1

                if it > iterations:
                    break

        if it == iterations - 1:
            print("Finished by {} steps".format(iterations))

        self.y_pred = fetches[self.y]
        

    """Get unscaled data and return unscaled predictions"""
    def predict(self, sess, x):
        y_act = [0] * len(x)
        x, y_act = self._transform(x, y_act)
        feed_dict = { self.x: x, self.y_act: y_act }
        y = sess.run(self.y, feed_dict)
        #print(y)
        return self._inverse_y(y).reshape(1, -1).tolist()[0]

    def print_results(self):
        print(str("b0 {:.4}, b1 {:.4}, err {}").format(self.b0, self.b1, []))

    def _transform(self, x_in, y_in):
        x_in = np.array(x_in).reshape(-1, 1)
        y_in = np.array(y_in).reshape(-1, 1)
        if not hasattr(self, "x_scaler"):
            print("New scaler")
            self.x_scaler = preprocessing.StandardScaler().fit(x_in)
            self.y_scaler = preprocessing.StandardScaler().fit(y_in)
        x_std = self.x_scaler.transform(x_in)
        y_std = self.y_scaler.transform(y_in)
        return x_std.reshape(1, -1).tolist()[0], y_std.reshape(1, -1).tolist()[0]

    def _inverse_y(self, y_std):
       return self.y_scaler.inverse_transform(y_std) 


