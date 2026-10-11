from models.ResNet import *
from models.VGGNet import *
from models.MultiTaskModel import build_multitask_backbone


_SUPPORTED = {
    "VGG11", "VGG13", "VGG16", "VGG19",
    "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
}


def LoadModel(modelname, color, num_classes_per_task):
    """
    根据模型名 + 每任务类别数，构建多任务模型。

    Args:
        modelname: 模型名
        color:     0 灰度 / 1 RGB
        num_classes_per_task: list[int]，单任务时是 [N]
    """
    if modelname not in _SUPPORTED:
        raise ValueError(
            f"Unsupported model: '{modelname}'. "
            f"Supported: {sorted(_SUPPORTED)}"
        )

    return build_multitask_backbone(modelname, color, num_classes_per_task)