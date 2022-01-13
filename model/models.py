
from .lenet5 import lenet5

def get_model(model_name):
    if(model_name == "lenet5"):
        return lenet5
    else:
        print("unknown model")