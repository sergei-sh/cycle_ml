
from collections import deque
import numpy as np
import itertools as itools
import tensorflow as tf
import tensorflow.python.debug as tfdbg

class BadArgumentError(ValueError):
    pass

class Model:
    def __init__(self):
        pass

    def train(self, *, data=None, max_runs, optimizer):
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
            #b0_init = 1 
            #b1_init = 1 

        with tf.Session() as sess:
            print("Starting round with {} inputs, b0 = {:.1f}, b1 = {:.1f}".format(data_len, b0_init, b1_init))
            x = tf.placeholder(tf.float32, [data_len], name="x")
            y = tf.placeholder(tf.float32, [data_len], name="y")
            b0 = tf.Variable([b0_init], dtype=tf.float32, trainable=True, name="b0")
            b1 = tf.Variable([b1_init], dtype=tf.float32, trainable=True, name="b1")
            #the model
            y = tf.add(tf.mul(x, b1), b0)
            y_act = tf.placeholder(tf.float32, [data_len], name="y_act")
            ERROR_MIN = 1e-6
            error = tf.sqrt(tf.clip_by_value((y - y_act) * (y - y_act), clip_value_min=ERROR_MIN, clip_value_max=1e+127))
            #train_step = tf.train.GradientDescentOptimizer(learning_rate=precision).minimize(error)
            train_step = optimizer.minimize(error)
            x_in = self.data.wafer_counts
            y_in = self.data.cycle_times
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            feed_dict = { x: self.data.wafer_counts, y_act: self.data.cycle_times }

            #writer = tf.summary.FileWriter("log_graph", sess.graph)
            #for value in [b0, b1, error]:
            #    tf.summary.tensor_summary(value.op.name, value)
            #summaries = tf.summary.merge_all()
            
            grad1 = tf.gradients(error, [b0])[0]
            grad2 = tf.gradients(error, [b1])[0]
            #out = tf.Print(error, [error])
            fetches_in = { b0: b0, b1: b1, y: y, error: error, train_step: train_step, grad1: grad1, grad2: grad2}#, summaries: summaries}

            #tf.logging.set_verbosity(tf.logging.DEBUG)
            tail = deque(itools.repeat(0, 8), maxlen=8)
            print(error.get_shape()[0])
            prev_err = np.zeros(error.get_shape()[0])
            for i in range(0, max_runs):
                fetches = sess.run(fetches_in, feed_dict)
                #writer.add_summary(fetches[summaries])

                self.b0 = fetches[b0][0]
                self.b1 = fetches[b1][0]
                #self.err = np.mean(fetches[error])
                self.err = fetches[error]
                if i % 100 == 0:
                    print("Step {}".format(i))
                    self.print_results()

                #print(fetches[grad1], fetches[grad2])
                delta_err = np.subtract(fetches[error], prev_err)
                prev_err = fetches[error]
                #print(np.ma.sum(delta_err))

                tail.append((self.b0, self.b1))
                tail_extr = list(tail)[1::2]
                if len(set(tail_extr)) == 1: #every other equals the last
                    print("Finished by oscillation, step {}".format(i))
                    break

                if np.isnan(self.b0) or np.isnan(self.b1):
                    raise ValueError("Variable is NaN!")

            if i == max_runs - 1:
                print("Finished by {} steps".format(max_runs))

            self.y_pred = fetches[y]
        self.print_results()         

    def print_results(self):
        print(str("b0 {:.4}, b1 {:.4}, err {}").format(self.b0, self.b1, []))


