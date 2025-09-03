from models.ResNet import *
from models.VGGNet import *

def LoadModel(modelname, color, numclasses):
    #VGGNet
    if modelname =="VGG11":
        model = VGG11(color=color, num_classes=numclasses, init_weights=False)
    elif modelname =="VGG13":
        model = VGG13(color=color, num_classes=numclasses, init_weights=False)  
    elif modelname =="VGG16":
        model = VGG16(color=color, num_classes=numclasses, init_weights=False)
    elif modelname =="VGG19":
        model = VGG19(color=color, num_classes=numclasses, init_weights=False) 
    #ResNet
    elif modelname =="ResNet18":
        model = ResNet18(color=color, num_classes=numclasses)   
    elif modelname =="ResNet34":
        model = ResNet34(color=color, num_classes=numclasses)   
    elif modelname =="ResNet50":
        model = ResNet50(color=color, num_classes=numclasses)   
    elif modelname =="ResNet101":
        model = ResNet101(color=color, num_classes=numclasses)   
    elif modelname =="ResNet152":
        model = ResNet152(color=color, num_classes=numclasses)   
    else:
        model = "unknow model:      " + modelname
    return model