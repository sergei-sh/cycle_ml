
from os.path import join

def _data_dir():
    return "models"

def _tr_fname(tool_recipe):    
    return tool_recipe.replace(",", "_")

def get_path(tool_recipe):
    return join(_data_dir(),  _tr_fname(tool_recipe) + ".tf")

def get_model_path(tool_recipe):
    return join(_data_dir(), _tr_fname(tool_recipe) + ".weights")
