import os
import argparse
import cv2
import numpy as np
import torch


class Predict():
    def __init__(self, args):
        # ---------- 设备 ----------
        if args.device == 'cpu' or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{args.device}')

        self.color = args.color
        self.imgsize = tuple(args.img_size)      # (W, H)

        # ---------- 类别名（可选） ----------
        self.classes = None
        if args.data and os.path.isfile(args.data):
            import yaml
            with open(args.data, 'r', encoding='utf-8') as f:
                msg = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.classes = msg.get('classes', None)

        # ---------- 加载模型 ----------
        obj = torch.load(args.weight, map_location='cpu')
        if isinstance(obj, dict) and 'state_dict' in obj:
            obj = obj['state_dict']

        if not isinstance(obj, torch.nn.Module):
            raise ValueError(
                f"{args.weight} 不是完整模型对象。"
                f"请确保 train.py 用 torch.save(model) 保存整模型。"
            )

        self.model = obj.to(self.device)
        self.model.eval()

    # -------------------------------------------------------------
    def imagepreprocess(self, image):
        image = cv2.resize(image, self.imgsize,
                           interpolation=cv2.INTER_AREA)

        if self.color == 0:
            # 灰度：(H, W) -> (1, 1, H, W)
            img = np.expand_dims(image, axis=0)
            img = np.expand_dims(img, axis=0)
        else:
            # BGR -> RGB，HWC -> CHW
            img = image[:, :, ::-1].transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)

        img = np.ascontiguousarray(img)
        return img

    # -------------------------------------------------------------
    @torch.inference_mode()
    def predicter(self, path):
        data = cv2.imread(path, self.color)
        if data is None:
            print(f"[跳过] 无法读取图片：{path}")
            return None

        img = self.imagepreprocess(data)
        img = torch.from_numpy(img).float()
        img = img.to(self.device, non_blocking=True) * 0.00390625   # /256

        pred = self.model(img)
        _, result = torch.max(pred, 1)
        idx = result.item()

        if self.classes is not None:
            inv = {v: k for k, v in self.classes.items()}
            return idx, inv.get(idx, f'unknown_{idx}')
        return idx


# -----------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument("--path", type=str,
                        default="./data/test",
                        help='待预测图片所在文件夹')
    parser.add_argument("--color", default=1, type=int,
                        help='0=灰度(1通道)，1=RGB(3通道)')
    parser.add_argument("--weight", default='./run/best.pth', type=str,
                        help='完整模型对象路径')
    parser.add_argument("--data", default='', type=str,
                        help='可选：训练用的 yaml，用于读取 classes')
    parser.add_argument("--img_size", default=[224, 224], type=int, nargs=2,
                        help='输入尺寸，如 --img_size 224 224')
    parser.add_argument("--device", default='0', type=str,
                        help='0/1/... 表示 GPU 编号，cpu 表示 CPU')
    args = parser.parse_args()

    T = Predict(args)

    path = args.path
    names = sorted(os.listdir(path))
    for name in names:
        p = os.path.join(path, name)
        if not os.path.isfile(p):
            continue
        out = T.predicter(p)
        if out is not None:
            print(f"{name}: {out}")