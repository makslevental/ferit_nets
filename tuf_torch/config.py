import os
import random
import datetime

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

TORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEBUG = False
PROJECT_ROOT = '/home/maksim/ferit_nets/'
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
BATCH_SIZE: int = 1024
SHUFFLE_DL = True
N_STRAT_SPLITS = 10
EPOCHS = 20

LR = 1e-3
OPTIMIZER = lambda net: torch.optim.Adam(net.parameters(), lr=LR)
SCHEDULER = lambda optimizer, dataloader: CosineAnnealingLR(optimizer, len(dataloader))
CRITERION = CrossEntropyLoss()
word_file = "/usr/share/dict/words"
WORDS = open(word_file).read().splitlines()
PROJECT_NAME = random.choice(WORDS)

word_file = "/usr/share/dict/words"
WORDS = open(word_file).read().splitlines()

print(PROJECT_NAME)

NETS_PATH = os.path.join(PROJECT_ROOT, "nets", f"{datetime.date.today()}_{PROJECT_NAME}")
FIGS_PATH = os.path.join(PROJECT_ROOT, "figs", f"{datetime.date.today()}_{PROJECT_NAME}")
LOGS_PATH = os.path.join(PROJECT_ROOT, "logs", f"{datetime.date.today()}_{PROJECT_NAME}")

if not os.path.exists(NETS_PATH):
    os.makedirs(NETS_PATH)
if not os.path.exists(FIGS_PATH):
    os.makedirs(FIGS_PATH)
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
