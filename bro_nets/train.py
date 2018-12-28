import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from bro_nets.config import TORCH_DEVICE, DATA_ROOT, BATCH_SIZE, SHUFFLE_DL, EPOCHS, OPTIMIZER, SCHEDULER, CRITERION
from bro_nets.cross_val import tuf_table_csv_to_df
from bro_nets.data import AlarmDataset
from bro_nets.models427 import GPR_15_300
from bro_nets.visualization import writer


def train(
        net: torch.nn.Module,
        dataloader: DataLoader,
        criterion,  # torch.nn._Loss
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epochs=10,
) -> torch.nn.Module:
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

    return net


if __name__ == '__main__':
    root = os.getcwd()
    tuf_table_file_name = 'small_maxs_table.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(root, tuf_table_file_name))
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
