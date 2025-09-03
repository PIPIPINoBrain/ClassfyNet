import torch.nn as nn
import torch

#都加了bn层，不加对一些数据集（相似度较高）不友好
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()


        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes))
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

def make_features(cfg: list, color: int):
    layers = []
    if color ==0:
        in_channels = 1
    else:
        in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
            bn = nn.BatchNorm2d(v)                   #+bn层了
            layers += [conv2d, bn, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

        

vgg_cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def VGG11(color=1, **kwargs):
    v11 = vgg_cfgs["vgg11"]
    model = VGG(make_features(v11, color), **kwargs)
    return model


def VGG13(color=1, **kwargs):
    v13 = vgg_cfgs["vgg13"]
    model = VGG(make_features(v13, color), **kwargs)
    return model


def VGG16(color=1, **kwargs):
    v16 = vgg_cfgs["vgg16"]
    model = VGG(make_features(v16, color), **kwargs)
    return model

def VGG19(color=1, **kwargs):
    v19 = vgg_cfgs["vgg19"]
    model = VGG(make_features(v19, color), **kwargs)
    return model


if  __name__=="__main__":
    net = VGG16(color =1, num_classes=5, init_weights=True)
    print(net)