import os

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
TORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEBUG = True
PROJECT_ROOT = '/home/maksim/dev_projects/ferit_nets/'
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
BATCH_SIZE: int = 128
SHUFFLE_DL = True
N_STRAT_SPLITS = 10
EPOCHS = 100

LR = 1e-3
OPTIMIZER = lambda net: torch.optim.Adam(net.parameters(), lr=LR)
SCHEDULER = lambda optimizer, dataloader: CosineAnnealingLR(optimizer, len(dataloader))
CRITERION = CrossEntropyLoss()
