import os
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from tuf_torch.config import TORCH_DEVICE, DATA_ROOT, BATCH_SIZE, SHUFFLE_DL, EPOCHS, OPTIMIZER, SCHEDULER, CRITERION, \
    PROJECT_ROOT
from tuf_torch.cross_val import tuf_table_csv_to_df
from tuf_torch.data import AlarmDataset
from tuf_torch.models427 import GPR_15_300
from tuf_torch.visualization import writer


def train(
        net: torch.nn.Module,
        dataloader: DataLoader,
        criterion,  # torch.nn._Loss
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epochs=10,
) -> Iterable[torch.nn.Module]:
    net.to(TORCH_DEVICE, dtype=torch.float)

    i = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        for j, (inputs, labels) in enumerate(dataloader):
            # get the inputs
            inputs, labels = (
                inputs.to(TORCH_DEVICE, dtype=torch.float),
                labels.to(TORCH_DEVICE, dtype=torch.long)
            )
            i += len(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if np.isnan(loss.item()):
                raise Exception('gradients blew up')
            writer.add_scalar('Train/Loss', loss, i)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], i)
        scheduler.step()
        yield net, loss.item()


if __name__ == '__main__':
    tuf_table_file_name = 'small_maxs_table.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, tuf_table_file_name))
    ad = AlarmDataset(all_alarms, DATA_ROOT)
    adl = DataLoader(ad, BATCH_SIZE, SHUFFLE_DL)
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
    # print(net.state_dict())
