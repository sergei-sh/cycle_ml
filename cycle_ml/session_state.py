""" 
Updated: 2017
Author: Sergei Shliakhtin
Contact: xxx.serj@gmail.com
Notes: TF save/load routines
"""

import tensorflow as tf

from cycle_ml.aux import get_model_path

def save(sess, tool_recipe):
    saver = tf.train.Saver()
    return saver.save(sess, get_model_path(tool_recipe))

def load(sess, tool_recipe):    
    path_name = get_model_path(tool_recipe) 
    saver = tf.train.Saver()#import_meta_graph(path_name + ".meta")
    print(saver.last_checkpoints)
    saver.restore(sess, path_name) 

