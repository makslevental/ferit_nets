import os

import torch
from torch.utils.data import DataLoader

from tuf_torch.config import DATA_ROOT, PROJECT_ROOT, BATCH_SIZE, SHUFFLE_DL, OPTIMIZER, SCHEDULER, CRITERION, EPOCHS
from tuf_torch.cross_val import tuf_table_csv_to_df
from tuf_torch.data import AlarmDataset
from tuf_torch.models427 import GPR_15_300
from tuf_torch.test import test
from tuf_torch.train import train
from tuf_torch.visualization import plot_roc

tuf_table_file_name = 'small_maxs_table.csv'
all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, tuf_table_file_name))
ad = AlarmDataset(
    all_alarms,
    DATA_ROOT,
)

adl = DataLoader(ad, BATCH_SIZE, SHUFFLE_DL, num_workers=4)

net = torch.nn.DataParallel(GPR_15_300())

optim = OPTIMIZER(net)
sched = SCHEDULER(optim, adl)
net = train(
    net,
    adl,
    criterion=CRITERION,
    optimizer=optim,
    scheduler=sched,
    epochs=EPOCHS
)


roc, auc, all_labels, confs = test(net, adl)

plot_roc(
    roc,
    auc,
    f'F1V4p4v3',
    show=True
)