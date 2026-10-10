from models.ResNet import *
from models.VGGNet import *


_SUPPORTED = {
    "VGG11", "VGG13", "VGG16", "VGG19",
    "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
}


def LoadModel(modelname, color, numclasses, zero_init_residual=False):
    """
    zero_init_residual 只对 ResNet 生效，VGG 会忽略。
    """
    if modelname not in _SUPPORTED:
        raise ValueError(
            f"Unsupported model: '{modelname}'. "
            f"Supported: {sorted(_SUPPORTED)}"
        )

    # ---------------- VGG ----------------
    if modelname == "VGG11":
        return VGG11(color=color, num_classes=numclasses, init_weights=False)
    elif modelname == "VGG13":
        return VGG13(color=color, num_classes=numclasses, init_weights=False)
    elif modelname == "VGG16":
        return VGG16(color=color, num_classes=numclasses, init_weights=False)
    elif modelname == "VGG19":
        return VGG19(color=color, num_classes=numclasses, init_weights=False)

    # ---------------- ResNet ----------------
    elif modelname == "ResNet18":
        return ResNet18(color=color, num_classes=numclasses,
                        zero_init_residual=zero_init_residual)
    elif modelname == "ResNet34":
        return ResNet34(color=color, num_classes=numclasses,
                        zero_init_residual=zero_init_residual)
    elif modelname == "ResNet50":
        return ResNet50(color=color, num_classes=numclasses,
                        zero_init_residual=zero_init_residual)
    elif modelname == "ResNet101":
        return ResNet101(color=color, num_classes=numclasses,
                         zero_init_residual=zero_init_residual)
    elif modelname == "ResNet152":
        return ResNet152(color=color, num_classes=numclasses,
                         zero_init_residual=zero_init_residual)