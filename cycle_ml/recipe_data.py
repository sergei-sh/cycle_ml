
import itertools as itools
import numpy as np
import tensorflow as tf

from cycle_ml.aux import get_path

_NONE = 0

class RecipeData:
    def __init__(self, wafer_counts = None, cycle_times = None, predicted = None):
       self.wafer_counts = wafer_counts if wafer_counts else []
       self.cycle_times = cycle_times if cycle_times else []
       self.predicted = predicted if predicted else [] 
       self.assert_len()
       self.wc_pending = _NONE
       self.predicted_pending = _NONE

    def save(self, tool_recipe):
        writer = tf.python_io.TFRecordWriter(get_path(tool_recipe))
        serialized = b""
        for i in range(0, len(self.wafer_counts)):
            xval = self.wafer_counts[i]
            yval = self.cycle_times[i]
            predval = self.predicted[i]
            example = tf.train.Example(
                features=tf.train.Features(
                  feature={
                    'x': tf.train.Feature(
                        float_list=tf.train.FloatList(value=[xval,])),
                    'y': tf.train.Feature(
                        float_list=tf.train.FloatList(value=[yval,])),
                    'pred': tf.train.Feature(
                        float_list=tf.train.FloatList(value=[predval,])),
                    }))
            serialized = example.SerializeToString()
            writer.write(serialized)
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


    def load(self, tool_recipe):
        tf_record_iterator = tf.python_io.tf_record_iterator(get_path(tool_recipe))
        try:
            for serialized_example in tf_record_iterator:
                example = tf.train.Example()
                example.ParseFromString(serialized_example)

                if 'x' in example.features.feature:
                    self.wafer_counts.extend(example.features.feature['x'].float_list.value)
                    self.cycle_times.extend(example.features.feature['y'].float_list.value)
                    self.predicted.extend(example.features.feature['pred'].float_list.value)
                    self.assert_len()
                else:
                    self.wc_pending = example.features.feature['x_pending'].float_list.value[0]
                    self.predicted_pending = example.features.feature['pred_pending'].float_list.value[0]
        except tf.errors.NotFoundError:
            pass

    def assert_len(self):
       assert len(self.wafer_counts) == len(self.cycle_times) == len(self.predicted)

    def __str__(self):
        msg = ""
        for i in range(0, len(self.wafer_counts)):
            msg += "{} {} {}; \n".format(self.wafer_counts[i], self.cycle_times[i], self.predicted[i])
        msg += "wc_pending: {} \n".format(self.wc_pending)
        msg += "prdicted_pending: {} \n".format(self.predicted_pending)
        return msg

    def __len__(self):
        return len(self.wafer_counts)

    def acquire_pending(self, ct_pending):
        """Cycle time needs a pair to be acquired with"""
        assert self.wc_pending
        self.wafer_counts.append(self.wc_pending)
        self.cycle_times.append(ct_pending)
        self.predicted.append(self.predicted_pending)
        self.assert_len()
        self.wc_pending = _NONE
        self.predicted_pending = _NONE

def mae10(recipe_data):
   TAIL = 10 
   pred = list(itools.dropwhile(lambda x: not x, recipe_data.predicted[-TAIL:]))
   if pred:
       diff = np.subtract(recipe_data.cycle_times[-len(pred):], pred)
       return np.mean(np.absolute(diff))
