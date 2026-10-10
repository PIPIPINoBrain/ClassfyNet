import os
import cv2
import numpy as np
import yaml
from torch.utils import data


class DatasetLoader(data.Dataset):
    def __init__(self, txt_path, classes, imgsize, color=0):
        """
        Args:
            txt_path: 数据集描述文件，每行 "<图像路径> <类别名>"
            classes:  dict, {类别名: label}，例如 {'JAPAN':0, 'JWAWA':1, 'MAL':2}
            imgsize:  [H, W]
            color:    0 -> 灰度(1通道)，1 -> RGB(3通道)
        """
        self.txt_path = txt_path
        self.classes = classes
        self.H = imgsize[0]
        self.W = imgsize[1]
        self.color = color

        self.images = []
        self.labels = []
        self.checklist = self._load_txt(txt_path)

        print(f"Datasets checked completed: {txt_path}")
        for key in self.checklist.keys():
            print(f"  {key}: {self.checklist[key]}")
        print()

    # -------------------------------------------------------------
    def _load_txt(self, txt_path):
        """逐行解析 txt，返回每个类别的样本数统计。"""
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"数据集描述文件不存在：{txt_path}")

        checklist = {name: 0 for name in self.classes.keys()}

        with open(txt_path, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # 从右侧分割一次：左边可能是带空格的路径，右边是类别名
                parts = line.rsplit(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"{txt_path} 第 {lineno} 行格式错误：'{line}'，"
                        f"应为 '<图像路径> <类别名>'"
                    )

                img_path, class_name = parts

                if class_name not in self.classes:
                    raise ValueError(
                        f"{txt_path} 第 {lineno} 行类别名 '{class_name}' "
                        f"不在 classes 中，合法类别：{list(self.classes.keys())}"
                    )

                if not os.path.isfile(img_path):
                    raise FileNotFoundError(
                        f"{txt_path} 第 {lineno} 行图像不存在：{img_path}"
                    )

                self.images.append(img_path)
                self.labels.append(self.classes[class_name])
                checklist[class_name] += 1

        if len(self.images) == 0:
            raise RuntimeError(f"{txt_path} 中没有任何有效样本")

        return checklist

    # -------------------------------------------------------------
    def imageresize(self, image):
        return cv2.resize(image, (self.W, self.H),
                          interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------------
    def __getitem__(self, i):
        img_p = self.images[i]
        label = self.labels[i]

        data = cv2.imread(img_p, self.color)
        if data is None:
            raise FileNotFoundError(f"无法读取图片：{img_p}")

        data = self.imageresize(data)

        if self.color == 0:
            # 灰度：(H, W) -> (1, H, W)
            img = np.expand_dims(data, axis=0)
        else:
            # BGR -> RGB，(H, W, C) -> (C, H, W)
            img = data[:, :, ::-1].transpose((2, 0, 1))

        img = np.ascontiguousarray(img)
        return img, label

    # -------------------------------------------------------------
    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    yaml_path = './data/demo.yaml'   # 改成你自己的 yaml 路径

    with open(yaml_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    classes = result['classes']
    color = result.get('color', 0)

    dataset = DatasetLoader(result['train'], classes, [224, 224], color)
    print(f"总样本数: {len(dataset)}")

    for i in range(min(3, len(dataset))):
        img, label = dataset[i]
        print(f"sample {i}: path={dataset.images[i]}, "
              f"label={label}, shape={img.shape}, dtype={img.dtype}")







