import os
import yaml
import math
import argparse
import numpy as np
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
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1



class Train():
    def __init__(self, args):
        with open(args.data, 'r', encoding='utf-8') as f:
            message = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.classes = message['classes']
        self.color = message['color']
        self.modelname = message['model']
        train_datasets = DatasetLoader(message['train'], self.classes, args.img_size, self.color)
        val_datasets = DatasetLoader(message['val'], self.classes, args.img_size, self.color)

        self.train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(dataset=val_datasets,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True)
        model = LoadModel(self.modelname, self.color, len(self.classes))
        print(model)
        if args.weight != '':
            model.load_state_dict(torch.load(args.pretrained_model))
        #导入开源的进行靠近训练，修改层级可以
        '''
        model = resnet50(num_classes=1000)
        model.load_state_dict(torch.load(args.pretrained_model))
        channel_in = model.fc.in_features  # 获取fc层的输入通道数
        model.fc = nn.Linear(channel_in, len(self.classes))
        '''
        self.device = select_device(args.device, batch_size=args.batch_size)
        self.model = model.to(self.device)
        self.lf = lambda x: (1 - x / args.epochs) * (1.0 - 0.01) + 0.01
        self.cost = nn.CrossEntropyLoss().to(self.device)
        # Optimization
        if args.optimizer=='Adam':
            self.optimizer = Adam(model.parameters(), lr=args.lr,  weight_decay=1e-8)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(model.parameters(), lr=args.lr, betas=args.momentum, weight_decay=1e-8)
        else:
            self.optimizer = SGD(model.parameters(), lr=args.lr,  momentum=args.momentum, nesterov=True)

    def val(self, model):
        model.eval()
        pbar_v = enumerate(self.val_loader)
        pbar_v = tqdm(pbar_v, total=len(self.val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        all_labels = 0
        correct = 0
        for i, (imgs, targets) in pbar_v:
            imgs = imgs.to(self.device, non_blocking=True).float() * 0.00390625
            targets = targets.to(self.device, non_blocking=True)
            preds = self.model(imgs)
            _, predications = torch.max(preds.data, 1)
            all_labels += targets.size(0)
            correct += (predications == targets).sum().item()
            acc = float(correct / all_labels)
            pbar_v.set_description(('Accuracy:  %10.4g') % (acc))
        return acc
    
    def trainer(self):
        nb = len(self.train_loader)
        best_fitness = -1
        for epoch in range(args.epochs):
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            self.optimizer.zero_grad()
            for i, (imgs, targets) in pbar:
                imgs = imgs.to(self.device, non_blocking=True).float()*0.00390625
                targets = targets.to(self.device, non_blocking=True)
                preds = self.model(imgs)
                loss = self.cost(preds, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('Epochs: %5s' + '     mems: %5s' + '     loss: %5.4g') % (
                    f'{epoch}/{args.epochs - 1}', mem, loss))
            acc = self.val(self.model)
            if (loss - acc) >best_fitness:
                torch.save(self.model, os.path.join(args.model_path, 'best.pth'))
            torch.save(self.model, os.path.join(args.model_path, 'last.pth'))

        torch.cuda.empty_cache()

        return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--data", default='./data/demo.yaml', type=str)
    parser.add_argument("--weight", default='', type=str)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--img_size", default=[224, 224], type=int)
    parser.add_argument("--device", default='0', type=str, help="cuda/cpu")
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument("--model_path", default='./run', type=str)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.97, type=float)
    args = parser.parse_args()

    T = Train(args)
    T.trainer()