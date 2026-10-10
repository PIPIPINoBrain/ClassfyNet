@西安燧仓科技有限公司

# 用于简单分类的网络：A network for simple classification

一个基于 PyTorch 的简单图像分类项目，支持 VGG 和 ResNet 系列，提供训练、测试和 ONNX 导出功能。

------

## 目前支持的网络：Currently supported networks

**VGG Net**

| name    | VGG11 | VGG13 | VGG16 | VGG19 |
| :------ | :---- | :---- | :---- | :---- |
| support | √     | √     | √     | √     |

**Res Net**

| name    | ResNet18 | ResNet34 | ResNet50 | ResNet101 | ResNet152 |
| :------ | :------- | :------- | :------- | :-------- | :-------- |
| support | √        | √        | √        | √         | √         |

------

## 环境依赖

text

```
python >= 3.8
torch >= 1.10
torchvision
opencv-python
numpy
pyyaml
tqdm
onnx        # 仅在导出 ONNX 时需要
```



安装：

bash

```
pip install torch torchvision opencv-python numpy pyyaml tqdm onnx
```



------

## 数据组织方式

数据集通过 **txt 文件**描述，每行格式为：

text

```
<图像路径> <类别名>
```



示例 `train.txt`：

text

```
E:/Dog/JAPAN/img001.jpg JAPAN
E:/Dog/JAPAN/img002.jpg JAPAN
E:/Dog/JWAWA/img001.jpg JWAWA
E:/Dog/MAL/img001.jpg   MAL
```



**要点：**

- 路径可以包含空格，解析时从**右侧最后一个空格**切分；
- 类别名必须与 yaml 里 `classes` 的 key 完全一致（大小写敏感）；
- `#` 开头的行和空行会被跳过，可写注释；
- 路径建议用正斜杠 `/`，避免 Windows 反斜杠转义问题。

------

## 配置文件（demo.yaml）

yaml

```
# train/val/test 指向描述数据集的 txt 文件
train: ./data/train.txt
val:   ./data/val.txt
test:  ./data/test.txt

# 模型名称，见上方支持列表
model: ResNet18

color: 0                  # 0 gray(1通道)  1 RGB(3通道)

classes:                  # 类别名 -> 类别号
  JAPAN: 0
  JWAWA: 1
  MAL:   2
```



------

## 训练：Train

bash

```
python train.py --data ./data/demo.yaml
```



### 训练参数

| 关键词         | 定义                                                         | 默认值             | 是否必须修改 |
| :------------- | :----------------------------------------------------------- | :----------------- | :----------- |
| `--data`       | 配置文件 demo.yaml 路径                                      | `./data/demo.yaml` | √            |
| `--weight`     | 预训练权重路径（`.pth` 整模型或 state_dict），留空则从头训练 | `''`               | ×            |
| `--epochs`     | 训练轮数                                                     | `150`              | ×            |
| `--batch_size` | 每次训练的批次大小                                           | `16`               | ×            |
| `--img_size`   | 模型输入尺寸，例如 `--img_size 224 224`                      | `[224, 224]`       | ×            |
| `--device`     | `0` / `1` / ... 表示 GPU 编号；`cpu` 表示 CPU                | `0`                | ×            |
| `--workers`    | DataLoader 线程数                                            | `4`                | ×            |
| `--model_path` | 训练结果保存目录                                             | `./run`            | ×            |
| `--optimizer`  | 优化器，可选 `SGD` / `Adam` / `AdamW`                        | `Adam`             | ×            |
| `--lr`         | 学习率                                                       | `1e-3`             | ×            |
| `--momentum`   | 动量（SGD 时）或 β1（Adam/AdamW 时）                         | `0.9`              | ×            |
| `--cos_lr`     | 使用 one-cycle cosine 调度；不加则使用 CosineAnnealingLR     | 关闭               | ×            |

### 训练输出

- `run/best.pth`：验证集准确率最高的模型（整模型对象）
- `run/last.pth`：最后一个 epoch 的模型（整模型对象）

### 示例

bash

```
# 最简
python train.py --data ./data/demo.yaml

# 使用 SGD，指定学习率
python train.py --data ./data/demo.yaml --optimizer SGD --lr 0.01

# 加载预训练权重微调
python train.py --data ./data/demo.yaml --weight ./run/best.pth
```



------

## 测试：Test

bash

```
python test.py --path ./data/test --weight ./run/best.pth --color 0
```



### 测试参数

| 关键词       | 定义                                               | 默认值           | 是否必须修改 |
| :----------- | :------------------------------------------------- | :--------------- | :----------- |
| `--path`     | 待测试图片所在文件夹（不支持单张图片）             | `./data/test`    | √            |
| `--color`    | `0` 灰度 / `1` RGB，**必须与训练时一致**           | `1`              | √            |
| `--weight`   | 训练好的模型路径                                   | `./run/best.pth` | √            |
| `--img_size` | 模型输入尺寸，例如 `--img_size 224 224`            | `[224, 224]`     | √            |
| `--data`     | 可选，训练用的 yaml，用于读取 `classes` 显示类别名 | `''`             | ×            |
| `--device`   | `0` / `1` / ... 或 `cpu`                           | `0`              | ×            |

### 示例

bash

```
# 基本用法
python test.py --path ./data/test \
               --color 0 \
               --weight ./run/best.pth \
               --img_size 224 224

# 显示类别名
python test.py --path ./data/test \
               --color 0 \
               --weight ./run/best.pth \
               --img_size 224 224 \
               --data ./data/demo.yaml

# 强制 CPU
python test.py --device cpu
```



输出示例：

text

```
img001.jpg: (0, 'JAPAN')
img002.jpg: (1, 'JWAWA')
img003.jpg: (2, 'MAL')
```



------

## 导出 ONNX

bash

```
python export.py --weight ./run/best.pth --input_size "1,1,224,224"
```



### 导出参数

| 关键词               | 定义                                               | 默认值           |
| :------------------- | :------------------------------------------------- | :--------------- |
| `--weight`           | 训练好的模型路径（整模型对象）                     | `./run/best.pth` |
| `--input_size`       | 输入尺寸 `batch,channel,H,W`，例如 `"1,3,224,224"` | `"1,3,224,224"`  |
| `--output`           | 输出 ONNX 路径，默认与 weight 同目录同名           | 自动             |
| `--device`           | `cpu` / `cuda` / `0` / `1` ...                     | `cpu`            |
| `--onnx_version`     | ONNX opset 版本                                    | `12`             |
| `--dynamic`          | 启用动态 batch / H / W                             | 关闭             |
| `--verbose`          | 打印导出详细日志                                   | 关闭             |
| `--constant_folding` | 启用常量折叠优化                                   | 关闭             |

### 示例

bash

```
# 灰度模型，224 输入
python export.py --weight ./run/best.pth --input_size "1,1,224,224"

# RGB 模型，动态 batch
python export.py --weight ./run/best.pth \
                 --input_size "1,3,224,224" \
                 --dynamic

# 指定输出路径
python export.py --weight ./run/best.pth \
                 --input_size "1,1,224,224" \
                 --output ./onnx/model_v1.onnx
```



**注意**：`--input_size` 的**通道数必须和模型一致**。

- 训练时 `color: 0` → 模型是灰度 → 用 `"1,1,224,224"`
- 训练时 `color: 1` → 模型是 RGB → 用 `"1,3,224,224"`

通道数不匹配会报类似 `expected input[1,3,224,224] to have 1 channels` 的错误。

------

## 项目结构

text

```
ClassfyNet/
├── train.py              # 训练入口
├── test.py               # 测试入口
├── export.py             # ONNX 导出
├── models/
│   ├── ResNet.py         # ResNet 系列模型定义
│   └── VGGNet.py         # VGG 系列模型定义
├── utils/
│   ├── utils.py          # LoadModel 等
│   ├── torch_utils.py    # select_device / EarlyStopping 等
│   └── dataset.py        # DatasetLoader
├── data/
│   ├── demo.yaml         # 配置文件
│   ├── train.txt         # 训练集描述
│   ├── val.txt           # 验证集描述
│   └── test.txt          # 测试集描述
└── run/                  # 训练输出目录
    ├── best.pth
    └── last.pth
```



------

## 常见问题

**Q1: 训练时 loss 不下降、acc 贴着随机水平？**

- 检查 `color` 和 txt 里的图片通道是否一致；
- 检查类别名是否和 yaml 的 `classes` 完全一致；
- 先用 10 张图做「过拟合测试」，理论上 loss 应迅速降到接近 0。如果 10 张都过拟合不了，是代码/数据 bug；能过拟合则是超参问题。

**Q2: `export.py` 报 `expected input[...] to have 1 channels`？**

模型是灰度（`color: 0`），但 `--input_size` 用了 3 通道。改成 `--input_size "1,1,224,224"`。

**Q3: `test.py` 报 `'dict' object has no attribute 'to'`？**

你加载的 `.pth` 是 `state_dict` 而非整模型对象。本项目 `train.py` 用 `torch.save(self.model, ...)` 保存整模型，请确保用同一套脚本训练出的权重。

**Q4: 训练集和验证集能是同一份吗？**

技术上可以，但**强烈不建议**。这样 `val acc` 就是训练集 acc，会虚高，`best.pth` 会保存过拟合的模型。请用两个不同的 txt 文件。

**Q5: 如何换其他输入尺寸？**

改 `--img_size`（命令行）和 yaml 里对应的尺寸即可。模型已使用 `AdaptiveAvgPool2d`，对输入尺寸有较好兼容性。VGG 因为分类头写死了 `512*7*7`，输入过小时特征图会塌陷，建议输入 ≥ 224。

------

## 引用与致谢

- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- 部分工具函数参考 [YOLOv5](https://github.com/ultralytics/yolov5)

------

## 更新记录

### v1.1（2026.10.10）

- 数据集从「按文件夹组织」改为「按 txt 描述文件组织」，路径与类别名更灵活；
- `train.py` 修复了 `args.pretrained_model` 未定义的 bug；
- 学习率调度真正启用（`CosineAnnealingLR`）；
- 保存 `best.pth` 的判据由 `loss - acc` 改为 `acc > best_acc`；
- `val` 阶段使用 `torch.inference_mode()`，节省显存；
- `export.py` 的 `--input_size` 支持从命令行解析；
- `test.py` 增加 `--device` / `--data` 参数，输出类别名。

### v1.2

- 初始版本，支持 VGG11/13/16/19 与 ResNet18/34/50/101/152；
- 支持训练、测试、ONNX 导出。
