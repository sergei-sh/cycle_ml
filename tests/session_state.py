""" 
Updated: 2017
Author: Sergei Shliakhtin
Contact: xxx.serj@gmail.com
Notes:
"""

import tensorflow as tf
import sys

PATH = "test_m/model"

def state_graph(*, save):
    val = [99] if save else [0]
    print("\n\ninit val: ", val)
    w = tf.Variable(val, dtype=tf.float32)    

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if save:
            saver = tf.train.Saver()
            saver.save(sess, PATH)
            print("saved")
        else:
            print(sess.run([w])[0])
            saver = tf.train.Saver()
            saver.restore(sess, PATH)
            print("w: ", sess.run([w])[0])

state_graph(save=True)            
#state_graph(save=False)            
