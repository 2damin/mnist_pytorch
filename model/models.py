
from model.vgg16 import VGG16
from .lenet5 import lenet5

def get_model(model_name):
    if(model_name == "lenet5"):
        return lenet5
    elif(model_name == "vgg16"):
        return VGG16
    else:
        print("unknown model")