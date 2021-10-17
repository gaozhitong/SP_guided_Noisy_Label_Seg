import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from lib.configs.parse_arg import args


def random_init(seed=0):
    if args.deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)