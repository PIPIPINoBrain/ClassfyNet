import os
import cv2
import numpy as np
import torch
from torch.utils import data


class DatasetLoader(data.Dataset):
    def __init__(self, txt_path, tasks, imgsize, color=0):
        """
        Args:
            txt_path: 数据集描述文件，每行 "<路径> <类别1> <类别2> ..."
            tasks:    dict[task_name -> dict[class_name -> label]]
                      单任务时 len(tasks) == 1
            imgsize:  [H, W]
            color:    0 -> 灰度(1通道)，1 -> RGB(3通道)
        """
        self.txt_path = txt_path
        self.tasks = tasks
        self.task_names = list(tasks.keys())           # 保序
        self.num_tasks = len(self.task_names)
        self.num_classes_per_task = [len(tasks[t]) for t in self.task_names]

        self.H = imgsize[0]
        self.W = imgsize[1]
        self.color = color

        self.images = []
        self.labels = []                                # list[list[int]]
        self.checklist = self._load_txt(txt_path)

        print(f"Datasets checked completed: {txt_path}")
        for t_name in self.task_names:
            print(f"  [{t_name}] {self.checklist[t_name]}")
        print()

    # -------------------------------------------------------------
    def _load_txt(self, txt_path):
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"数据集描述文件不存在：{txt_path}")

        checklist = {t: {c: 0 for c in self.tasks[t].keys()}
                     for t in self.task_names}

        with open(txt_path, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.rsplit(maxsplit=self.num_tasks)
                if len(parts) != self.num_tasks + 1:
                    raise ValueError(
                        f"{txt_path} 第 {lineno} 行格式错误：'{line}'，"
                        f"应为 '<路径> " +
                        " ".join(f"<{t}>" for t in self.task_names) + "'"
                    )

                img_path = parts[0]
                class_names = parts[1:]

                if not os.path.isfile(img_path):
                    raise FileNotFoundError(
                        f"{txt_path} 第 {lineno} 行图像不存在：{img_path}"
                    )

                labels = []
                for t_idx, t_name in enumerate(self.task_names):
                    cname = class_names[t_idx]
                    if cname not in self.tasks[t_name]:
                        raise ValueError(
                            f"{txt_path} 第 {lineno} 行任务 '{t_name}' 的类别 "
                            f"'{cname}' 不在 yaml 中，"
                            f"合法类别：{list(self.tasks[t_name].keys())}"
                        )
                    labels.append(self.tasks[t_name][cname])
                    checklist[t_name][cname] += 1

                self.images.append(img_path)
                self.labels.append(labels)

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
        labels = self.labels[i]

        data = cv2.imread(img_p, self.color)
        if data is None:
            raise FileNotFoundError(f"无法读取图片：{img_p}")

        data = self.imageresize(data)

        if self.color == 0:
            img = np.expand_dims(data, axis=0)
        else:
            img = data[:, :, ::-1].transpose((2, 0, 1))

        img = np.ascontiguousarray(img)
        return img, torch.tensor(labels, dtype=torch.long)  # (T,)

    # -------------------------------------------------------------
    def __len__(self):                                       # ← 新增
        return len(self.images)