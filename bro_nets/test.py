import os

import torch

from bro_nets.cross_val import tuf_table_csv_to_df
from bro_nets.data import create_dataloader
from bro_nets.train import train
from bro_nets import device


def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test images: %d %%' % (
            100 * correct / total))


if __name__ == '__main__':
    root = os.getcwd()
    tuf_table_file_name = 'three_stratified_cross_val.csv'
    all_alarms_motherfucker = tuf_table_csv_to_df(os.path.join(root, 'data', tuf_table_file_name))

    dl = create_dataloader(all_alarms_motherfucker, batch_size=2)
    net = train(dl, 1000)
    test(net, dl)
