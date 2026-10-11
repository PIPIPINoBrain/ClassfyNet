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
        self.args = args

        # ---------- 读取配置 ----------
        with open(args.data, 'r', encoding='utf-8') as f:
            message = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.color = message['color']
        self.modelname = message['model']
        self.tasks = message['tasks']                        # {task: {cname: id}}
        self.task_names = list(self.tasks.keys())            # 保序
        self.num_tasks = len(self.task_names)
        self.num_classes_per_task = [len(self.tasks[t])
                                     for t in self.task_names]
        self.task_weights = message.get('task_weights', {})  # 可选

        print(f'任务数: {self.num_tasks}')
        for i, t in enumerate(self.task_names):
            print(f'  {t}: {self.num_classes_per_task[i]} 类')

        # ---------- 校验 best_metric ----------
        if args.best_metric not in ('mean', 'min') and \
                args.best_metric not in self.task_names:
            raise ValueError(
                f"--best_metric='{args.best_metric}' 无效。"
                f"可选：'mean' / 'min' / {self.task_names}"
            )

        # ---------- 数据集 ----------
        train_datasets = DatasetLoader(message['train'], self.tasks,
                                       args.img_size, self.color)
        val_datasets = DatasetLoader(message['val'], self.tasks,
                                     args.img_size, self.color)

        self.train_loader = torch.utils.data.DataLoader(
            train_datasets, batch_size=args.batch_size,
            num_workers=args.workers, shuffle=True, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
            val_datasets, batch_size=args.batch_size,
            num_workers=args.workers, shuffle=False, pin_memory=True)

        # ---------- 模型 ----------
        model = LoadModel(self.modelname, self.color,
                          num_classes_per_task=self.num_classes_per_task)
        print(model)

        # ---------- 预训练权重 ----------
        if args.weight and os.path.isfile(args.weight):
            self._load_pretrained(model, args.weight)

        # ---------- 设备 ----------
        self.device = select_device(args.device, batch_size=args.batch_size)
        self.model = model.to(self.device)

        # ---------- 损失 ----------
        self.costs = [nn.CrossEntropyLoss().to(self.device)
                      for _ in self.task_names]

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
    def _load_pretrained(self, model, path):
        """
        支持三种权重：
          1) 本项目 MultiTaskModel 的 state_dict（key 带 backbone./heads. 前缀）
          2) 单任务 ResNet state_dict（key 带 fc.*）
          3) 单任务 VGG state_dict（key 带 classifier.7.*）
        只加载 backbone 部分，heads 随机初始化。
        """
        state = torch.load(path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        elif not isinstance(state, dict):
            state = state.state_dict()

        has_backbone_prefix = any(k.startswith('backbone.') for k in state)

        if has_backbone_prefix:
            # 已经是 MultiTaskModel 的 state_dict
            missing, unexpected = model.load_state_dict(state, strict=False)
        else:
            # 单任务权重，映射到 backbone.*
            new_state = {}
            for k, v in state.items():
                if k.startswith('fc.'):               # ResNet 分类头
                    continue
                if k.startswith('classifier.7.'):     # VGG 最后一层 Linear
                    continue
                new_state[f'backbone.{k}'] = v
            missing, unexpected = model.load_state_dict(new_state, strict=False)

        print(f'Loaded weights from {path} | '
              f'missing={len(missing)}, unexpected={len(unexpected)}')
        if missing:
            print(f'  missing 前 5 个: {missing[:5]}')
        if unexpected:
            print(f'  unexpected 前 5 个: {unexpected[:5]}')

        # 兜底诊断
        if len(missing) == len(model.state_dict()):
            print('[警告] 全部 key 都没匹配，等于没加载预训练权重！')

    # ---------------------------------------------------------------
    @torch.inference_mode()
    def val(self, model):
        model.eval()
        correct = [0] * self.num_tasks
        total = 0

        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (imgs, targets) in pbar:
            imgs = imgs.to(self.device, non_blocking=True).float() * 0.00390625
            targets = targets.to(self.device, non_blocking=True)   # (B, T)

            preds = model(imgs)                                    # list of (B, n_i)
            if isinstance(preds, torch.Tensor):                    # 单任务兜底
                preds = [preds]

            for t in range(self.num_tasks):
                _, p = torch.max(preds[t], 1)
                correct[t] += (p == targets[:, t]).sum().item()
            total += targets.size(0)

            accs = [c / max(total, 1) for c in correct]
            pbar.set_description(
                ' | '.join(f'{self.task_names[t]}: {accs[t]:.4f}'
                           for t in range(self.num_tasks)))

        return [c / max(total, 1) for c in correct]

    # ---------------------------------------------------------------
    def trainer(self):
        args = self.args
        nb = len(self.train_loader)
        best_score = -1.0
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

                preds = self.model(imgs)                  # list of (B, n_i)
                if isinstance(preds, torch.Tensor):       # 单任务兜底
                    preds = [preds]

                # ---------- 加权多任务 loss ----------
                loss = 0.0
                for t in range(self.num_tasks):
                    w = self.task_weights.get(self.task_names[t], 1.0)
                    loss = loss + w * self.costs[t](preds[t], targets[:, t])

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(
                    f'Epoch {epoch}/{args.epochs - 1}  mem {mem}  '
                    f'loss {running_loss / (i + 1):.4f}  '
                    f'lr {self.optimizer.param_groups[0]["lr"]:.3g}')

            # ---------- 验证 ----------
            accs = self.val(self.model)                   # list[float]
            self.scheduler.step()

            # ---------- best 判据 ----------
            if args.best_metric == 'mean':
                score = sum(accs) / len(accs)
            elif args.best_metric == 'min':
                score = min(accs)
            else:
                t_idx = self.task_names.index(args.best_metric)
                score = accs[t_idx]

            # ---------- 保存 ----------
            torch.save(self.model, os.path.join(args.model_path, 'last.pth'))
            if score > best_score:
                best_score = score
                torch.save(self.model,
                           os.path.join(args.model_path, 'best.pth'))
                acc_str = ' | '.join(
                    f'{self.task_names[t]}: {accs[t]:.4f}'
                    for t in range(self.num_tasks))
                print(f'\n[Epoch {epoch}] New best '
                      f'({args.best_metric}={score:.4f})  {acc_str}')

        torch.cuda.empty_cache()
        return best_score


# -------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--data", default='./data/demo.yaml', type=str)
    parser.add_argument("--weight", default='', type=str,
                        help='预训练权重路径（单任务 ResNet state_dict '
                             '或本项目 MultiTaskModel）')
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--img_size", default=[224, 224], type=int, nargs=2,
                        help='输入尺寸，例如 --img_size 224 224')
    parser.add_argument("--device", default='0', type=str)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument("--model_path", default='./run', type=str)
    parser.add_argument('--optimizer', type=str,
                        choices=['SGD', 'Adam', 'AdamW'], default='Adam')
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--cos_lr", action='store_true',
                        help='使用 one-cycle cosine 调度；不加则用 CosineAnnealingLR')
    parser.add_argument("--best_metric", default='mean', type=str,
                        help="best 判据：'mean' / 'min' / <任务名>")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    T = Train(args)
    best = T.trainer()
    print(f'\n训练结束，best_score = {best:.4f}')
    print(f'模型保存在: {args.model_path}/best.pth')