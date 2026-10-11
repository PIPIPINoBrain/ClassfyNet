- ## 目前支持的网络

  **VGG Net**

  | name    | VGG11 | VGG13 | VGG16 | VGG19 |
  | :------ | :---- | :---- | :---- | :---- |
  | support | √     | √     | √     | √     |

  **Res Net**

  | name    | ResNet18 | ResNet34 | ResNet50 | ResNet101 | ResNet152 |
  | :------ | :------- | :------- | :------- | :-------- | :-------- |
  | support | √        | √        | √        | √         | √         |

  ------

  ## 核心概念：多任务

  本项目**所有任务统一走 `tasks` 格式**，没有单任务/多任务的分支：

  - **单任务**：`tasks` 里只有一个 key
  - **多任务**：`tasks` 里有多个 key，每个 key 是一个独立的分类任务

  模型结构 = **共享 backbone** + **多个独立分类头**（每个 head 输出对应任务的 logits）。

  text

  ```
  输入图像
      ↓
  [共享 Backbone]  (ResNet/VGG)
      ↓
    feature
      ↓           ↓            ↓
   [head0]     [head1]      [head2]
   shape       color        size
  ```

  

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
  onnx        # 仅导出 ONNX 时需要
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
  <图像路径> <任务1类别> <任务2类别> ...
  ```

  

  **任务数量 = yaml 里 `tasks` 的 key 数量，标签顺序必须和 yaml 一致。**

  ### 多任务示例（2 个任务：classes + types）

  text

  ```
  F:/data/RT/Positive/00000956.png    dog     painting
  F:/data/Small/Positive/00000977.png person  photo
  F:/data/Small/Negative/00001435.png cat     photo
  F:/data/Small/Negative/00000911.png person  painting
  ```

  

  ### 单任务示例（1 个任务）

  text

  ```
  E:/Dog/JAPAN/img001.jpg JAPAN
  E:/Dog/JWAWA/img001.jpg JWAWA
  E:/Dog/MAL/img001.jpg   MAL
  ```

  

  **要点：**

  - 路径可以包含空格，解析时从**右侧切 `num_tasks` 个字段**；
  - 类别名必须与 yaml 里对应任务的 `classes` key 完全一致（大小写敏感）；
  - `#` 开头的行和空行会被跳过，可写注释；
  - 路径建议用正斜杠 `/`，避免 Windows 反斜杠转义问题。

  ------

  ## 配置文件（demo.yaml）

  yaml

  ```
  train: ./data/train.txt
  val:   ./data/val.txt
  test:  ./data/test.txt
  
  # VGG11 VGG13 VGG16 VGG19
  # ResNet18 ResNet34 ResNet50 ResNet101 ResNet152
  model: ResNet18
  
  color: 0                       # 0 gray(1通道)  1 RGB(3通道)
  
  # ============ 多任务定义 ============
  # 顺序 = txt 每行标签的顺序
  tasks:
    classes:
      cat: 0
      dog: 1
      person: 2
    types:
      photo: 0
      painting: 1
  
  # 可选：每任务损失权重，默认全 1.0
  task_weights:
    classes: 1.0
    types: 1.0
  ```

  

  **单任务写法**（`tasks` 里只保留一个 key）：

  yaml

  ```
  tasks:
    classes:
      cat: 0
      dog: 1
      person: 2
  ```

  

  ------

  ## 训练：Train

  bash

  ```
  python train.py --data ./data/demo.yaml
  ```

  

  ### 训练参数

  | 关键词          | 定义                                              | 默认值             | 是否必须修改 |
  | :-------------- | :------------------------------------------------ | :----------------- | :----------- |
  | `--data`        | 配置文件 yaml 路径                                | `./data/demo.yaml` | √            |
  | `--weight`      | 预训练权重路径，留空则从头训练                    | `''`               | ×            |
  | `--epochs`      | 训练轮数                                          | `150`              | ×            |
  | `--batch_size`  | 每批大小                                          | `16`               | ×            |
  | `--img_size`    | 模型输入尺寸，`--img_size 224 224`                | `[224, 224]`       | ×            |
  | `--device`      | `0` / `1` / ... 或 `cpu`                          | `0`                | ×            |
  | `--workers`     | DataLoader 线程数                                 | `4`                | ×            |
  | `--model_path`  | 训练结果保存目录                                  | `./run`            | ×            |
  | `--optimizer`   | `SGD` / `Adam` / `AdamW`                          | `Adam`             | ×            |
  | `--lr`          | 学习率                                            | `1e-3`             | ×            |
  | `--momentum`    | SGD 动量 / Adam β1                                | `0.9`              | ×            |
  | `--cos_lr`      | 使用 one-cycle cosine；不加则用 CosineAnnealingLR | 关闭               | ×            |
  | `--best_metric` | best 判据：`mean` / `min` / `<任务名>`            | `mean`             | ×            |

  ### `--best_metric` 说明

  - `mean`：所有任务 acc 的平均值（默认）
  - `min`：所有任务 acc 中的最小值（逼迫模型不能放弃任何任务）
  - `<任务名>`：只关心某一个任务，比如 `--best_metric classes`

  ### 训练输出

  - `run/best.pth`：`best_metric` 最高的模型（整模型对象）
  - `run/last.pth`：最后一个 epoch 的模型（整模型对象）

  ### 预训练权重兼容

  `_load_pretrained` 支持三种权重格式，会自动映射到 `backbone.*`：

  1. **本项目 MultiTaskModel**（key 带 `backbone.` / `heads.` 前缀）→ 直接加载
  2. **单任务 ResNet**（key 带 `fc.*`）→ 丢弃 `fc.*`，其余映射到 `backbone.*`
  3. **单任务 VGG**（key 带 `classifier.7.*`）→ 丢弃 `classifier.7.*`，其余映射到 `backbone.*`

  加载后会打印 `missing` / `unexpected` 数量，方便排查：

  text

  ```
  Loaded weights from ./run_single/best.pth | missing=4, unexpected=2
    missing 前 5 个: ['heads.0.weight', 'heads.0.bias', 'heads.1.weight', 'heads.1.bias']
    unexpected 前 5 个: ['backbone.fc.weight', 'backbone.fc.bias']
  ```

  

  - `missing` 只包含 `heads.*` → **backbone 全部加载上了**（正常）
  - `missing` 里出现 `backbone.conv1.*` → **权重没加载上**（需排查）

  ### 示例

  bash

  ```
  # 多任务训练
  python train.py --data ./data/demo.yaml --epochs 100
  
  # 从单任务 ResNet 权重迁移
  python train.py --data ./data/demo.yaml --weight ./run_single/best.pth
  
  # 只关心 classes 任务
  python train.py --data ./data/demo.yaml --best_metric classes
  
  # 取最差任务 acc 作为 best 判据
  python train.py --data ./data/demo.yaml --best_metric min
  ```

  

  ------

  ## 测试：Test

  从 txt 读取测试集，支持**两种行格式**：

  - 只有路径 → 只输出预测
  - 路径 + 标签 → 输出预测 + 计算准确率

  bash

  ```
  python test.py \
      --yaml ./data/demo.yaml \
      --data ./data/test.txt \
      --weight ./run/best.pth \
      --color 0 \
      --img_size 224 224
  ```

  

  ### 测试参数

  | 关键词       | 定义                                   | 默认值           | 是否必须修改 |
  | :----------- | :------------------------------------- | :--------------- | :----------- |
  | `--yaml`     | 训练用的 yaml（读 `tasks`）            | 无               | √            |
  | `--data`     | 测试集 txt 文件                        | 无               | √            |
  | `--weight`   | 训练好的模型                           | `./run/best.pth` | √            |
  | `--color`    | `0` 灰度 / `1` RGB，**必须与训练一致** | `0`              | √            |
  | `--img_size` | 输入尺寸                               | `[224, 224]`     | √            |
  | `--device`   | `0` / `1` / ... 或 `cpu`               | `0`              | ×            |

  ### 输出示例

  **多任务，混合模式（前 3 行带标签，最后 1 行只有路径）：**

  text

  ```
  任务列表: ['classes', 'types']
  
  00000956.png: classes: pred=dog gt=dog ✓ | types: pred=painting gt=painting ✓
  00000977.png: classes: pred=person gt=person ✓ | types: pred=photo gt=photo ✓
  00001435.png: classes: pred=dog gt=cat ✗ | types: pred=photo gt=photo ✓
  unknown.jpg: classes: cat | types: photo
  
  ============================================================
  带标签样本数: 3
  [classes] 准确率: 0.6667 (2/3)
  [types] 准确率: 1.0000 (3/3)
  ```

  

  **单任务：**

  text

  ```
  任务列表: ['classes']
  
  img001.jpg: classes: pred=cat gt=cat ✓
  img002.jpg: classes: pred=dog gt=dog ✓
  img003.jpg: classes: pred=cat gt=dog ✗
  whatever.jpg: classes: cat
  
  ============================================================
  带标签样本数: 3
  [classes] 准确率: 0.6667 (2/3)
  ```

  

  ------

  ## 导出 ONNX

  bash

  ```
  python export.py --weight ./run/best.pth --input_size "1,1,224,224"
  ```

  

  ### 导出参数

  | 关键词               | 定义                                     | 默认值           |
  | :------------------- | :--------------------------------------- | :--------------- |
  | `--weight`           | 训练好的模型路径（整模型对象）           | `./run/best.pth` |
  | `--input_size`       | 输入尺寸 `batch,channel,H,W`             | `"1,1,224,224"`  |
  | `--output`           | 输出 ONNX 路径，默认与 weight 同目录同名 | 自动             |
  | `--device`           | `cpu` / `cuda` / `0` / `1` ...           | `cpu`            |
  | `--onnx_version`     | ONNX opset 版本                          | `12`             |
  | `--dynamic`          | 启用动态 batch / H / W                   | 关闭             |
  | `--verbose`          | 打印导出详细日志                         | 关闭             |
  | `--constant_folding` | 启用常量折叠优化                         | 关闭             |

  ### 多任务输出

  `export.py` 会自动检测 `MultiTaskModel.heads` 的数量，导出**多个输出**：

  text

  ```
  模型输出数: 2，输出名: ['output0', 'output1']
  ONNX 模型校验通过。
  Successfully exported ONNX model to: ./run/best.onnx
  ```

  

  **输出顺序 = yaml 里 `tasks` 的 key 顺序**：

  - `output0` → 第一个任务的 logits（`shape: (B, num_classes_0)`）
  - `output1` → 第二个任务的 logits（`shape: (B, num_classes_1)`）

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

  - 训练时 `color: 0` → 灰度 → 用 `"1,1,224,224"`
  - 训练时 `color: 1` → RGB → 用 `"1,3,224,224"`

  通道数不匹配会报：

  text

  ```
  RuntimeError: Given groups=1, weight of size [64, 1, 7, 7],
  expected input[1, 3, 224, 224] to have 1 channels, but got 3 channels instead
  ```

  

  ------

  ## 项目结构

  text

  ```
  ClassfyNet/
  ├── train.py                  # 训练入口
  ├── test.py                   # 测试入口
  ├── export.py                 # ONNX 导出
  ├── models/
  │   ├── ResNet.py             # ResNet 系列模型定义
  │   ├── VGGNet.py             # VGG 系列模型定义
  │   └── MultiTaskModel.py     # 多任务模型包装 + build_multitask_backbone
  ├── utils/
  │   ├── utils.py              # LoadModel
  │   ├── torch_utils.py        # select_device / EarlyStopping 等
  │   └── dataset.py            # DatasetLoader（多任务）
  ├── data/
  │   ├── demo.yaml             # 配置文件
  │   ├── train.txt             # 训练集描述
  │   ├── val.txt               # 验证集描述
  │   └── test.txt              # 测试集描述
  └── run/                      # 训练输出目录
      ├── best.pth
      └── last.pth
  ```

  

  ------

  ## 常见问题

  **Q1: `DataLoader` 报 `object of type 'DatasetLoader' has no len()`？**

  `DatasetLoader` 里缺少 `__len__` 方法。在类里加：

  python

  ```
  def __len__(self):
      return len(self.images)
  ```

  

  **Q2: `export.py` 报 `NameError: name 'output_path' is not defined`？**

  `export.py` 里输出路径的计算段被删了。加回：

  python

  ```
  if args.output:
      output_path = args.output
  else:
      base = os.path.splitext(args.weight)[0]
      output_path = base + ".onnx"
  ```

  

  **Q3: `export.py` 报 `expected input[...] to have 1 channels`？**

  模型是灰度（`color: 0`），但 `--input_size` 用了 3 通道。改成 `--input_size "1,1,224,224"`。

  **Q4: `test.py` 报 `AttributeError: 'Namespace' object has no attribute 'txt'`？**

  这是旧版 `test.py` 的 bug，主循环里误用了 `args.txt`。应该改成 `args.data`（测试集 txt）。当前版本已修复。

  **Q5: 训练时 loss 不下降、acc 贴着随机水平？**

  - 检查 `color` 和 txt 里图片通道是否一致；
  - 检查每行标签顺序是否和 yaml 里 `tasks` 的 key 顺序一致；
  - 先用 10 张图做「过拟合测试」，理论上 loss 应迅速降到接近 0。如果 10 张都过拟合不了，是代码/数据 bug；能过拟合则是超参问题。

  **Q6: 加载预训练权重后，`missing` 数量很多？**

  看 `missing` 前几个。如果只有 `heads.*`，正常（多任务头是新的）；如果包含 `backbone.*`，说明映射失败，检查权重文件格式。

  **Q7: VGG 加载单任务权重时，`unexpected` 里有 `classifier.7.\*`？**

  正常。VGG 多任务的 backbone 把 `classifier[-1]` 切掉了，单任务权重里对应的 `classifier.7.*` 会自动被忽略，不影响 backbone 加载。

  **Q8: 训练集和验证集能是同一份吗？**

  技术上可以，但**强烈不建议**。这样 `val acc` 就是训练集 acc，会虚高，`best.pth` 会保存过拟合的模型。请用两个不同的 txt 文件。

  **Q9: 如何换其他输入尺寸？**

  改 `--img_size` 即可。模型已使用 `AdaptiveAvgPool2d`，对输入尺寸有较好兼容性。VGG 因为分类头写死了 `512*7*7`，输入过小时特征图会塌陷，建议输入 ≥ 224。

  ------

  ## 引用与致谢

  - ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  - VGG: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
  - 部分工具函数参考 [YOLOv5](https://github.com/ultralytics/yolov5)

  ------

  ## 更新记录

  ### v2.0（多任务版本）

  - 项目重构为**多任务分类**架构：共享 backbone + 多个独立分类头
  - 新增 `models/MultiTaskModel.py`：`MultiTaskModel` + `build_multitask_backbone`
  - 数据集格式从 `<路径> <类别>` 改为 `<路径> <类别1> <类别2> ...`，任务数由 yaml 的 `tasks` 决定
  - yaml 的 `classes` 字段改为 `tasks`（支持单任务：`tasks` 只含一个 key）
  - 新增 `task_weights`：按任务加权 loss
  - 新增 `--best_metric`：`mean` / `min` / `<任务名>`
  - `test.py` 改为**从 txt 读取**，支持「只有路径」和「路径+标签」混合格式，自动计算准确率
  - `export.py` 支持**多输出 ONNX**（输出数 = `model.heads` 的数量）
  - `train.py` 的 `_load_pretrained` 支持三种权重格式（MultiTask / 单任务 ResNet / 单任务 VGG）
  - `_load_pretrained` 打印 `missing` / `unexpected` 前 5 个 key，便于排查

  ### v1.1

  - 数据集从「按文件夹组织」改为「按 txt 描述文件组织」
  - 修复 `args.pretrained_model` 未定义的 bug
  - 学习率调度真正启用（`CosineAnnealingLR`）
  - 保存 best 的判据由 `loss - acc` 改为 `acc > best_acc`
  - `val` 阶段使用 `torch.inference_mode()`
  - `export.py` 的 `--input_size` 支持从命令行解析

  ### v1.0

  - 初始版本，支持 VGG11/13/16/19 与 ResNet18/34/50/101/152
  - 支持训练、测试、ONNX 导出
