import os
import argparse
import cv2
import numpy as np
import torch
import yaml


class Predict():
    def __init__(self, args):
        # ---------- 设备 ----------
        if args.device == 'cpu' or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{args.device}')

        self.color = args.color
        self.imgsize = tuple(args.img_size)

        # ---------- 读 tasks ----------
        with open(args.yaml, 'r', encoding='utf-8') as f:
            msg = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.tasks = msg['tasks']
        self.task_names = list(self.tasks.keys())
        self.num_tasks = len(self.task_names)

        # ---------- 模型 ----------
        obj = torch.load(args.weight, map_location='cpu')
        if isinstance(obj, dict) and 'state_dict' in obj:
            obj = obj['state_dict']
        if not isinstance(obj, torch.nn.Module):
            raise ValueError(f"{args.weight} 不是完整模型对象。")

        self.model = obj.to(self.device).eval()

    # -------------------------------------------------------------
    def imagepreprocess(self, image):
        image = cv2.resize(image, self.imgsize,
                           interpolation=cv2.INTER_AREA)
        if self.color == 0:
            img = np.expand_dims(image, axis=0)
            img = np.expand_dims(img, axis=0)
        else:
            img = image[:, :, ::-1].transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)

    # -------------------------------------------------------------
    @torch.inference_mode()
    def predicter(self, path):
        data = cv2.imread(path, self.color)
        if data is None:
            return None

        img = self.imagepreprocess(data)
        img = torch.from_numpy(img).float().to(self.device) * 0.00390625

        preds = self.model(img)                       # list of (1, n_i)

        result = {}
        for t_idx, t_name in enumerate(self.task_names):
            _, p = torch.max(preds[t_idx], 1)
            idx = p.item()
            inv = {v: k for k, v in self.tasks[t_name].items()}
            result[t_name] = (idx, inv.get(idx, f'unknown_{idx}'))
        return result

    # -------------------------------------------------------------
    def parse_line(self, line):
        """
        解析一行，返回 (img_path, label_names | None)

        - 只有路径 → (path, None)
        - 路径 + N 个标签 → (path, [label1, ..., labelN])
        - 空行 / 注释 → None

        解析策略：从右侧切 N 个字段，检查它们是否都是合法类别名。
        是 → 认为是标签；不是 → 整行当作路径。
        """
        line = line.strip()
        if not line or line.startswith('#'):
            return None

        parts = line.rsplit(maxsplit=self.num_tasks)
        if len(parts) == 1:
            return parts[0], None

        img_path = parts[0]
        label_names = parts[1:]

        # 检查后半部分是否都是合法类别名
        for t_idx, t_name in enumerate(self.task_names):
            if label_names[t_idx] not in self.tasks[t_name]:
                # 不是标签 → 整行是路径（路径里有空格）
                return line, None

        return img_path, label_names


# -----------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument("--yaml", required=True, type=str,
                        help='训练用的 yaml（必需，用于读取 tasks）')
    parser.add_argument("--data", required=True, type=str,
                        help='测试集描述文件，每行 "<路径> [<类别1> ...]"')
    parser.add_argument("--color", default=0, type=int)
    parser.add_argument("--weight", default='./run/best.pth', type=str)
    parser.add_argument("--img_size", default=[224, 224], type=int, nargs=2)
    parser.add_argument("--device", default='0', type=str)
    args = parser.parse_args()

    T = Predict(args)

    # ---------- 统计 ----------
    total_with_label = 0
    correct_per_task = [0] * T.num_tasks
    failed_imgs = 0

    print(f"任务列表: {T.task_names}\n")

    with open(args.data, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            parsed = T.parse_line(line)
            if parsed is None:
                continue
            img_path, label_names = parsed

            if not os.path.isfile(img_path):
                print(f"[第 {lineno} 行] 图像不存在：{img_path}")
                failed_imgs += 1
                continue

            out = T.predicter(img_path)
            if out is None:
                print(f"[第 {lineno} 行] 无法读取：{img_path}")
                failed_imgs += 1
                continue

            name = os.path.basename(img_path)

            if label_names is None:
                # ---------- 只有路径：只输出预测 ----------
                pred_str = " | ".join(
                    f"{t}: {out[t][1]}" for t in T.task_names
                )
                print(f"{name}: {pred_str}")
            else:
                # ---------- 有标签：输出预测 + 判定 ----------
                total_with_label += 1
                details = []
                for t_idx, t_name in enumerate(T.task_names):
                    pred_idx, pred_name = out[t_name]
                    gt_name = label_names[t_idx]
                    gt_idx = T.tasks[t_name].get(gt_name, -1)
                    ok = (pred_idx == gt_idx)
                    if ok:
                        correct_per_task[t_idx] += 1
                    mark = '✓' if ok else '✗'
                    details.append(
                        f"{t_name}: pred={pred_name} gt={gt_name} {mark}"
                    )
                print(f"{name}: " + " | ".join(details))

    # ---------- 汇总 ----------
    print()
    print("=" * 60)
    if total_with_label > 0:
        print(f"带标签样本数: {total_with_label}")
        for t_idx, t_name in enumerate(T.task_names):
            acc = correct_per_task[t_idx] / total_with_label
            print(f"[{t_name}] 准确率: {acc:.4f} "
                  f"({correct_per_task[t_idx]}/{total_with_label})")
    else:
        print("测试集没有标签，只输出预测结果，无法计算准确率。")
    if failed_imgs > 0:
        print(f"失败图像数: {failed_imgs}")