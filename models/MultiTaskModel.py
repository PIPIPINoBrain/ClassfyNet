import torch
import torch.nn as nn


class MultiTaskModel(nn.Module):
    """
    共享 backbone + 多个独立分类头。
    backbone 输出 feature vector，每个 head 输出对应任务的 logits。
    """

    def __init__(self, backbone, feature_dim, num_classes_per_task):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_tasks = len(num_classes_per_task)

        self.heads = nn.ModuleList([
            nn.Linear(feature_dim, n) for n in num_classes_per_task
        ])

    def forward(self, x):
        feat = self.backbone(x)                        # (B, feature_dim)
        return [head(feat) for head in self.heads]     # list of (B, n_i)


def build_multitask_backbone(modelname, color, num_classes_per_task):
    """
    构建 backbone + 多 head。
    支持 ResNet 系列和 VGG 系列。
    """

    # ================== ResNet ==================
    if modelname in ("ResNet18", "ResNet34", "ResNet50",
                     "ResNet101", "ResNet152"):
        from models.ResNet import (
            ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
        )
        if modelname == "ResNet18":
            m = ResNet18(color=color, num_classes=2)
        elif modelname == "ResNet34":
            m = ResNet34(color=color, num_classes=2)
        elif modelname == "ResNet50":
            m = ResNet50(color=color, num_classes=2)
        elif modelname == "ResNet101":
            m = ResNet101(color=color, num_classes=2)
        else:
            m = ResNet152(color=color, num_classes=2)

        feature_dim = m.fc.in_features                 # 512 * expansion
        m.fc = nn.Identity()                           # 丢掉 fc
        return MultiTaskModel(m, feature_dim, num_classes_per_task)

    # ================== VGG ==================
    elif modelname in ("VGG11", "VGG13", "VGG16", "VGG19"):
        from models.VGGNet import VGG11, VGG13, VGG16, VGG19
        if modelname == "VGG11":
            m = VGG11(color=color, num_classes=2)
        elif modelname == "VGG13":
            m = VGG13(color=color, num_classes=2)
        elif modelname == "VGG16":
            m = VGG16(color=color, num_classes=2)
        else:
            m = VGG19(color=color, num_classes=2)

        # classifier 结构: Dropout, Linear(25088→4096), ReLU,
        #                  Dropout, Linear(4096→4096), ReLU, Dropout,
        #                  Linear(4096→num_classes)
        # classifier[:-1] 去掉最后的 Linear，输出 4096 维
        feature_dim = 4096
        m.classifier = m.classifier[:-1]
        return MultiTaskModel(m, feature_dim, num_classes_per_task)

    else:
        raise ValueError(f"MultiTask 暂不支持模型：{modelname}")