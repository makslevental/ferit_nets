import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from bro_nets.cross_val import tuf_table_csv_to_df
from bro_nets.data import create_dataloader
from bro_nets.models427 import GPR_15_300
from bro_nets.visualization import writer


def train(dataloader: DataLoader, epochs=10):
    net = GPR_15_300()
    # net.cuda()
    # net.train()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net.to(device, dtype=torch.float)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    i = 0
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for inputs, labels in dataloader:
            # get the inputs
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)

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

if __name__ == '__main__':
    root = os.getcwd()
    tuf_table_file_name = 'three_stratified_cross_val.csv'
    all_alarms_motherfucker = tuf_table_csv_to_df(os.path.join(root, 'data', tuf_table_file_name))

    dl = create_dataloader(all_alarms_motherfucker)
    train(dl)
