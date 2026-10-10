import os
import yaml
import math
import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW
from tqdm import tqdm
from utils.utils import *
from utils.torch_utils import *
from utils.dataset import DatasetLoader

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """cosine ramp from y1 to y2"""
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


class Train():
    def __init__(self, args):
        # ---------- 读取配置 ----------
        with open(args.data, 'r', encoding='utf-8') as f:
            message = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.classes = message['classes']
        self.color = message['color']
        self.modelname = message['model']

        # ---------- 数据集 ----------
        train_datasets = DatasetLoader(message['train'], self.classes,
                                       args.img_size, self.color)
        val_datasets = DatasetLoader(message['val'], self.classes,
                                     args.img_size, self.color)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_datasets,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=True,
            pin_memory=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=val_datasets,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,          # 验证集不 shuffle
            pin_memory=True,
        )

        # ---------- 模型 ----------
        model = LoadModel(self.modelname, self.color, len(self.classes))
        print(model)

        # ---------- 预训练权重加载 ----------
        if args.weight and os.path.isfile(args.weight):
            state = torch.load(args.weight, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            elif not isinstance(state, dict):
                state = state.state_dict()

            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f'Loaded weights from {args.weight} | '
                  f'missing={len(missing)}, unexpected={len(unexpected)}')
            if len(missing) == len(model.state_dict()):
                print('[警告] 全部 key 都没匹配，等于没加载预训练权重！')

        # ---------- 设备 ----------
        self.device = select_device(args.device, batch_size=args.batch_size)
        self.model = model.to(self.device)

        # ---------- 损失 ----------
        self.cost = nn.CrossEntropyLoss().to(self.device)

        # ---------- 优化器 ----------
        if args.optimizer == 'Adam':
            self.optimizer = Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-8)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(model.parameters(), lr=args.lr,
                                   betas=(args.momentum, 0.999),
                                   weight_decay=1e-8)
        else:  # SGD
            self.optimizer = SGD(model.parameters(), lr=args.lr,
                                 momentum=args.momentum, nesterov=True)

        # ---------- 学习率调度 ----------
        if args.cos_lr:
            self.lf = one_cycle(1, 0.01, args.epochs)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=self.lf)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epochs, eta_min=1e-5)

    # ---------------------------------------------------------------
    @torch.inference_mode()
    def val(self, model):
        model.eval()
        pbar_v = tqdm(enumerate(self.val_loader),
                      total=len(self.val_loader),
                      bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        all_labels = 0
        correct = 0
        for i, (imgs, targets) in pbar_v:
            imgs = imgs.to(self.device, non_blocking=True).float() * 0.00390625
            targets = targets.to(self.device, non_blocking=True)

            preds = self.model(imgs)
            _, predictions = torch.max(preds, 1)

            all_labels += targets.size(0)
            correct += (predictions == targets).sum().item()
            acc = float(correct / all_labels)
            pbar_v.set_description(('Accuracy:  %10.4g') % acc)

        return acc

    # ---------------------------------------------------------------
    def trainer(self):
        nb = len(self.train_loader)
        best_acc = 0.0
        os.makedirs(args.model_path, exist_ok=True)

        for epoch in range(args.epochs):
            self.model.train()

            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)

            pbar = tqdm(enumerate(self.train_loader), total=nb,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

            running_loss = 0.0
            for i, (imgs, targets) in pbar:
                imgs = imgs.to(self.device, non_blocking=True).float() * 0.00390625
                targets = targets.to(self.device, non_blocking=True)

                preds = self.model(imgs)
                loss = self.cost(preds, targets)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(
                    ('Epochs: %5s     mems: %5s     loss: %5.4g     lr: %5.3g') % (
                        f'{epoch}/{args.epochs - 1}',
                        mem,
                        running_loss / (i + 1),
                        self.optimizer.param_groups[0]['lr'],
                    )
                )

            # ---------- 验证 ----------
            acc = self.val(self.model)
            self.scheduler.step()

            # ---------- 保存 last ----------
            torch.save(self.model, os.path.join(args.model_path, 'last.pth'))

            # ---------- 保存 best ----------
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model, os.path.join(args.model_path, 'best.pth'))
                print(f'\n[Epoch {epoch}] New best acc: {best_acc:.4f}, '
                      f'saved best.pth')

        torch.cuda.empty_cache()
        return best_acc


# -------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--data", default='./data/demo.yaml', type=str)
    parser.add_argument("--weight", default='', type=str,
                        help='预训练权重路径（state_dict 或整个模型都可）')
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--img_size", default=[224, 224], type=int, nargs=2,
                        help='输入图像尺寸，例如 224 224')
    parser.add_argument("--device", default='0', type=str,
                        help="0/1/... 表示 GPU 编号，cpu 表示 CPU")
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader 的 worker 数')
    parser.add_argument("--model_path", default='./run', type=str)
    parser.add_argument('--optimizer', type=str,
                        choices=['SGD', 'Adam', 'AdamW'], default='SGD')
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--cos_lr", action='store_true',
                        help='使用 cosine 学习率衰减；不加则用 CosineAnnealingLR')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    T = Train(args)
    T.trainer()