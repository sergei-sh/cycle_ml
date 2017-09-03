""" 
Updated: 2017
Author: Sergei Shliakhtin
Contact: xxx.serj@gmail.com
Notes: Project observations/predictions data storage
"""

import itertools as itools
import numpy as np
import tensorflow as tf

from cycle_ml.aux import get_path, log_verbose


class RecipeData:
    """Saves data into TF binary format"""

    MAE_TAIL = 10 
    _NONE = 0

    def __init__(self, wafer_counts = None, cycle_times = None, abs_err = None):
       self.wafer_counts = wafer_counts if wafer_counts else []
       self.cycle_times = cycle_times if cycle_times else []
       self.abs_err = abs_err if abs_err else [] 
       self.assert_len()
       self.wc_pending = self._NONE
       self.predicted_pending = self._NONE

    def clear_batch(self):
        self.wafer_counts[:] = []       
        self.cycle_times[:] = []

    def save(self, tool_recipe):
        writer = tf.python_io.TFRecordWriter(get_path(tool_recipe))
        serialized = b""
        """batch data"""
        for i in range(0, len(self.wafer_counts)):
            xval = self.wafer_counts[i]
            yval = self.cycle_times[i]
            example = tf.train.Example(
                features=tf.train.Features(
                  feature={
                    'x': tf.train.Feature(
                        float_list=tf.train.FloatList(value=[xval,])),
                    'y': tf.train.Feature(
                        float_list=tf.train.FloatList(value=[yval,])),
                    }))
            serialized = example.SerializeToString()
            writer.write(serialized)
        """pending values"""
        if self.wc_pending:
            example = tf.train.Example(
                features=tf.train.Features(
                  feature={
                        'x_pending': tf.train.Feature(
                            float_list=tf.train.FloatList(value=[float(self.wc_pending,)])),
                        'pred_pending': tf.train.Feature(
                            float_list=tf.train.FloatList(value=[float(self.predicted_pending,)])),

                    }))
            serialized = example.SerializeToString()
            writer.write(serialized)
        """absolute error values"""
        self.abs_err[:] = self.abs_err[-self.MAE_TAIL:]
        example = tf.train.Example(
            features=tf.train.Features(
              feature={
                    'abs_err': tf.train.Feature(
                        float_list=tf.train.FloatList(value=self.abs_err)),
                }))
        serialized = example.SerializeToString()
        writer.write(serialized)


    def load(self, tool_recipe):
        tf_record_iterator = tf.python_io.tf_record_iterator(get_path(tool_recipe))
        try:
            for serialized_example in tf_record_iterator:
                example = tf.train.Example()
                example.ParseFromString(serialized_example)
                if 'x' in example.features.feature:
                    rec_wcount = example.features.feature['x'].float_list.value
                    assert 1 == len(rec_wcount)
                    self.wafer_counts.append(rec_wcount[0])
                    rec_ctyme = example.features.feature['y'].float_list.value
                    assert 1 == len(rec_ctyme)
                    self.cycle_times.append(rec_ctyme[0])
                    self.assert_len()
                elif 'abs_err' in example.features.feature:
                    self.abs_err = example.features.feature['abs_err'].float_list.value
                else:
                    self.wc_pending = example.features.feature['x_pending'].float_list.value[0]
                    self.predicted_pending = example.features.feature['pred_pending'].float_list.value[0]

        except tf.errors.NotFoundError:
            pass

    def assert_len(self):
        """Checks class invariant"""    
        assert len(self.wafer_counts) == len(self.cycle_times)

    def __str__(self):
        msg = ""
        for i in range(0, len(self.wafer_counts)):
            msg += "{} {}; \n".format(self.wafer_counts[i], self.cycle_times[i] )
        msg += "wc_pending: {} \n".format(self.wc_pending)
        msg += "prdicted_pending: {} \n".format(self.predicted_pending)
        msg += "abs_err {} \n".format(self.abs_err)
        return msg

    def __len__(self):
        return len(self.wafer_counts)

    def acquire_pending(self, ct_pending):
        """Implements next_datapoint/finish_datapoint architecture; a pending wafer_count is assumed to be stored 
        as wc_pending in this recipe data, when loading on finish_datapoint; the newly coming cycle_time (ct_pending)
        is paired with this wafer count and stored in normal storage"""
        assert self.wc_pending
        self.wafer_counts.append(self.wc_pending)
        self.cycle_times.append(ct_pending)
        """Having real and predicted cycle times, update error values"""
        last_err = abs(ct_pending - self.predicted_pending)
        self.abs_err.append(last_err)
        self.assert_len()
        self.wc_pending = self._NONE
        self.predicted_pending = self._NONE
        return last_err

def mae10(recipe_data):
   """Comppute Mean Absolute Error using last 10 abs.errs""" 
   pred = recipe_data.abs_err
   if pred:
       return np.mean(pred)
   else:
       return -1
