import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from bro_nets.cross_val import tuf_table_csv_to_df, region_and_stratified
from bro_nets.data import create_dataloader
from bro_nets.models427 import GPR_15_300
from bro_nets.visualization import writer
from bro_nets import device


def train(dataloader: DataLoader, epochs=10):
    net = GPR_15_300()

    print(device)

    net.to(device, dtype=torch.float)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    i = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for inputs, labels in dataloader:
            # get the inputs
            inputs, labels = (
                inputs.to(device, dtype=torch.float),
                labels.to(device, dtype=torch.long)
            )

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            i += 1
            writer.add_scalar('Train/Loss', loss, i)
            if i % 200 == 199:  # print every 200 mini-batches (mini batch is batch inside multi epoch?)
                print(f'[epoch {epoch}, batch {i}] loss: {running_loss:.10f}')
            # running_loss = 0.0

    print('Finished Training')
    writer.close()
    return net


def train_with_cross_val(alarms: pd.DataFrame):
    for train_split, test_data in region_and_stratified(alarms, n_splits_region=3, n_splits_stratified=5):
        test_loader = create_dataloader(test_data)
        for inputs, labels in test_loader:
            print(inputs, labels)
        for train_data, validation_data in train_split:
            train_loader = create_dataloader(train_data)
            validation_loader = create_dataloader(validation_data)

            for inputs, labels in train_loader:
                print(inputs, labels)
                print(validation_data)
            for inputs, labels in validation_loader:
                print(inputs, labels)
                print(validation_data)



if __name__ == '__main__':
    root = os.getcwd()
    tuf_table_file_name = 'small_maxs_table.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(root, tuf_table_file_name))

    train_with_cross_val(all_alarms)