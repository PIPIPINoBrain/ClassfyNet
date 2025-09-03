## 用于简单分类的网络：A network for simple classification

### 目前支持的网络：**Currently supported networks**（it will support more）

**VGG Net**

|  name   | VGG11 | VGG13 | VGG16 | VGG19 |
| :-----: | :---: | :---: | :---: | :---: |
| support |   √   |   √   |   √   |   √   |

**Res Net**

|  name   | ResNet18 | ResNet34 | ResNet50 | ResNet101 | ResNet152 |
| :-----: | :------: | :------: | :------: | :-------: | :-------: |
| support |    √     |    √     |    √     |     √     |     √     |

### 训练与测试: Train and Test

#### 训练：Train

**数据存储方式（以猫，狗，人分类为例，子文件夹命名需对应目标类别）**

train

——cat（images files）

——dog（images files）

——person（images files）

val

——cat（images files）

——dog（images files）

——person（images files）

test

——cat（images files）

——dog（images files）

——person（images files）



**数据文件设置（demo.yaml）**

train: 训练集路径

val:验证集路径

test:测试集路径

model：训练模型名称（目前支持的网络名称）

color：是否为灰度图，关乎于输入通道（0：灰度图，1：RGB）

classes：分类类别（训练集中子文件夹名：类别号）

​	cat：0

​	dog：1

​	person：2



**训练参数设置（train.py）**

| 关键词     | 定义                                   | 是否必须修改 |
| ---------- | -------------------------------------- | ------------ |
| data       | 数据文件路径 demo.yaml                 | √            |
| weight     | 预训练权重路径，不填写则无预训练权重   | ×            |
| epochs     | 训练轮数                               | ×            |
| batch_size | 每次训练的批次                         | ×            |
| img_size   | 模型输入尺寸                           | ×            |
| device     | 0 或者 cpu  目前只支持单gpu训练        | ×            |
| workers    | 线程数                                 | ×            |
| model_path | 模型训练结果路径                       | ×            |
| optimizer  | 反向传播迭代优化算法（SGD Adam AdamW） | ×            |
| lr         | 学习率                                 | ×            |
| momentum   | 动量                                   | ×            |

```python
python train.py
```



#### 测试：Test

**测试参数设置（test.py）**

| 关键词   | 定义                                                         | 是否必须修改 |
| -------- | ------------------------------------------------------------ | ------------ |
| path     | 测试文件夹路径（只文件夹路径测试，不支持单张图片，可自行改代码） | √            |
| color    | 是否为灰度图，关乎于输入通道（0：灰度图，1：RGB）            | √            |
| weight   | 训练好的权重                                                 | √            |
| img_size | 模型输入尺寸                                                 | √            |

```python
python test.py
```



#### 导出Onnx

```python
python export.py
```











