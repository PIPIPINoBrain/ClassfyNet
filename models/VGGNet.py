import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))          # ← 新增
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)                                   # ← 新增
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):               # ← 新增
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list, color: int):
    layers = []
    in_channels = 1 if color == 0 else 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1,
                               padding=1, bias=False)         # ← bias=False
            bn = nn.BatchNorm2d(v)
            layers += [conv2d, bn, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


vgg_cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def VGG11(color=1, **kwargs):
    return VGG(make_features(vgg_cfgs["vgg11"], color), **kwargs)


def VGG13(color=1, **kwargs):
    return VGG(make_features(vgg_cfgs["vgg13"], color), **kwargs)


def VGG16(color=1, **kwargs):
    return VGG(make_features(vgg_cfgs["vgg16"], color), **kwargs)


def VGG19(color=1, **kwargs):
    return VGG(make_features(vgg_cfgs["vgg19"], color), **kwargs)


if __name__ == "__main__":
    net = VGG16(color=1, num_classes=5, init_weights=True)
    print(net)