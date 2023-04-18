import numpy as np
import torch
import random
import os
import time


# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)  # 设置CPU随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为特定GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)  # 固定Numpy产生的随机数
    random.seed(seed)  # 设置整个Python基础环境中的随机种子
    if seed == 0:
        torch.backends.cudnn.benchmark = False  # 不使用选择卷积算法的机制，使用固定的卷积算法（可能会降低性能）
        torch.backends.cudnn.deterministic = True   # 只限制benchmark的确定性
        # torch.use_deterministic_algorithms(True)  # 避免所有不确定的原子操作，保证得到一样的结果
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


# data loader 的可重复性
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()