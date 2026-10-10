import os
import logging
import torch
import torch.distributed as dist
from contextlib import contextmanager

LOGGER = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    让所有进程等待 local_master 完成某些操作。
    用法：
        with torch_distributed_zero_first(local_rank):
            ...
    """
    if local_rank not in [-1, 0]:
        dist.barrier()
    try:
        yield
    finally:
        if local_rank == 0:
            dist.barrier()


def select_device(device='', batch_size=0, newline=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'Net 🚀'
    device = str(device).strip().lower().replace('cuda:', '')
    cpu = device == 'cpu'

    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and \
            torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, " \
            f"use '--device cpu' or pass valid CUDA device(s)"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else ['0']       # ← 改成列表
        n = len(devices)
        if n > 1 and batch_size > 0:
            assert batch_size % n == 0, \
                f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} " \
                 f"({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"
    else:
        s += 'CPU\n'

    if not newline:
        s = s.rstrip()
    print(s)                                                    # ← 补上打印
    return torch.device('cuda:0' if cuda else 'cpu')


class EarlyStopping:
    """简单的早停器（参考 YOLOv5）。"""

    def __init__(self, patience=30):
        self.best_fitness = float('-inf')                       # ← -inf
        self.best_epoch = 0
        self.patience = patience or float('inf')
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness

        delta = epoch - self.best_epoch
        self.possible_stop = delta >= (self.patience - 1)
        stop = delta >= self.patience

        if stop:
            LOGGER.info(
                f'Stopping training early as no improvement observed in last '
                f'{self.patience} epochs. Best results observed at epoch '
                f'{self.best_epoch}, best model saved as best.pth.'
            )
        return stop