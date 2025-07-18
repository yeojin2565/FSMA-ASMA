from .ECAPATDNN import *
from .rawnet3 import *
from .xvector import *
from .dvector import *

def create_model(model_name, num_classes, in_channels):
    if model_name == 'ECAPATDNN':
        model = ECAPA_TDNN(C=64,n_class=num_classes)
    elif model_name == 'rawnet3':
        model = RawNet3(C=128,nOut=num_classes)
    elif model_name == 'xvector':
        model = xvector(num_classes=num_classes)
    elif model_name == 'dvector':
        model = dvector(num_classes=num_classes)
    return model
