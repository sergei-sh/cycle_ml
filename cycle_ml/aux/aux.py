
from os.path import join, isfile
import time

def _data_dir():
    return "models"

def _tr_fname(tool_recipe):    
    return tool_recipe.replace(",", "_").replace("/", "_")

def get_path(tool_recipe):
    return join(_data_dir(), _tr_fname(tool_recipe) + ".tf")

def get_model_path(tool_recipe):
    return join(_data_dir(), _tr_fname(tool_recipe) + ".weights")

def model_exists(tool_recipe):
    return isfile(get_model_path(tool_recipe) + ".meta")    

LOG_DIRECTORY = "logs"    

def log_tool(tool_recipe, *, the_first, wafer_count, predicted, actual, abs_err, mae10, sequence):    
    fpath = join(LOG_DIRECTORY, _tr_fname(tool_recipe) + ".csv")
    time_str = time.strftime("%b %d %H:%M:%S %Y")
    with open(fpath, "w" if the_first else "a") as f:
        header = "DateTime, WaferCount, Predicted, Actual, AbsError, MeanAbsError(10), SequenceNo"
        message = "{}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:d}".format(
            time_str, wafer_count, predicted, actual, abs_err, mae10, sequence)
        if the_first:
            f.write(header)
        f.write("\n" + message)
        print(header)
        print(message)

log_verbose = False
