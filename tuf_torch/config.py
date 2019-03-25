import datetime
import os
import random
import tkinter as tk
from tkinter import simpledialog

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

TORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEBUG = False
PROJECT_ROOT = '/home/maksim/dev_projects/ferit_nets/'
DATA_ROOT = os.path.join(PROJECT_ROOT, 'F1V4p4v3')
BATCH_SIZE: int = 4096
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

application_window = tk.Tk()
application_window.withdraw()
PROJECT_NAME = simpledialog.askstring("Project name", "",
                                      parent=application_window, initialvalue=PROJECT_NAME)
if PROJECT_NAME is not None:
    project_path = f"{datetime.date.today()}_{PROJECT_NAME}"
    print(f"project path: {project_path}")
    NETS_PATH = os.path.join(PROJECT_ROOT, "nets", project_path)
    FIGS_PATH = os.path.join(PROJECT_ROOT, "figs", project_path)
    LOGS_PATH = os.path.join(PROJECT_ROOT, "logs", project_path)
else:
    raise Exception("no project name")

if not os.path.exists(NETS_PATH):
    os.makedirs(NETS_PATH)
if not os.path.exists(FIGS_PATH):
    os.makedirs(FIGS_PATH)
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
