import ast
import os
import time
from collections import defaultdict
from operator import itemgetter
import matplotlib.pyplot as plt

import torch
from bro_nets.models427 import GPR_15_300
from bro_nets.cross_val import tuf_table_csv_to_df, region_and_stratified, kfold_stratified_target_splits
from bro_nets.data import create_dataloader
from bro_nets.train import train
from bro_nets import TORCH_DEVICE
from sklearn.metrics import roc_curve, roc_auc_score
import torch.nn.functional as torch_f
import numpy as np

from bro_nets.visualization import plot_roc
from bro_nets.util import *

from scipy.stats.mstats import gmean


def test(net, testloader):
    all_labels = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(TORCH_DEVICE, dtype=torch.float), labels.to(TORCH_DEVICE, dtype=torch.long)
            outputs = net(inputs)
            conf = torch_f.softmax(outputs)[:, 1]
            all_labels = np.append(all_labels, labels)
            confs = np.append(confs, conf)

    roc = roc_curve(all_labels, confs)
    auc = roc_auc_score(all_labels, confs)
    return roc, auc, (all_labels, confs)


def test_fold_0():
    root = '/home/maksim/ferit_nets'
    start = time.time()

    tuf_table_file_name = 'fold_0_train.csv'
    train_alarms = tuf_table_csv_to_df(os.path.join(root, 'folds', tuf_table_file_name))
    train_alarms['utm'] = train_alarms['utm'].map(ast.literal_eval)
    tuf_table_file_name = 'fold_0_test.csv'
    test_alarms = tuf_table_csv_to_df(os.path.join(root, 'folds', tuf_table_file_name))
    test_alarms['utm'] = test_alarms['utm'].map(ast.literal_eval)

    train_splits, _, _, _ = kfold_stratified_target_splits(train_alarms, n_splits=10)
    nets = []
    for j, (train_data, validation_data) in enumerate(train_splits):
        train_dl = create_dataloader(train_data, os.path.join(root, 'data'), batch_size=128)
        validation_set_dl = create_dataloader(validation_data, os.path.join(root, 'data'), batch_size=128,
                                              m=train_dl.dataset._mean, s=train_dl.dataset._std)
        prescreener_roc = roc_curve(validation_data['HIT'], validation_data['conf'])
        prescreener_auc = roc_auc_score(validation_data['HIT'], validation_data['conf'])

        net_name = f'fold {j}'

        print(net_name)
        net = train(train_dl, 0)
        net_roc, net_auc, _ = test(net, validation_set_dl)
        plot_roc(
            [('net', net_roc), ('prescreener', prescreener_roc)],
            [net_auc, prescreener_auc],
            f'before train roc fold {j}'
        )

        net = train(train_dl, 1)
        net_roc, net_auc, _ = test(net, validation_set_dl)
        if .49 < net_auc < .51:
            continue
        plot_roc(
            [('net', net_roc), ('prescreener', prescreener_roc)],
            [net_auc, prescreener_auc],
            f'after train roc fold {j}'
        )
        nets.append(net)

    all_test_dl = create_dataloader(train_alarms, os.path.join(root, 'data'), batch_size=128)
    region_test_dl = create_dataloader(test_alarms, os.path.join(root, 'data'), batch_size=128,
                                       m=all_test_dl.dataset._mean, s=all_test_dl.dataset._std)

    prescreener_roc = roc_curve(test_alarms['HIT'], test_alarms['conf'])
    prescreener_auc = roc_auc_score(test_alarms['HIT'], test_alarms['conf'])
    # netrocs_netaucs_labelsconfs = map(lambda net: test(net, region_test_dl), nets)
    netrocs_netaucs_labelsconfs = []
    for i, net in enumerate(nets):
        roc, auc, labels_confs = test(net, region_test_dl)
        netrocs_netaucs_labelsconfs.append((roc, auc, labels_confs))
        plot_roc(
            [('net', roc),
             ('prescreener', prescreener_roc)],
            [auc, prescreener_auc],
            f'holdout roc fold {i}'
        )
        torch.save(net.state_dict(), f'net_fold_{i}.pth')
    confs = map(lambda x: x[2][1], netrocs_netaucs_labelsconfs)
    gm = np.mean(np.sort(np.stack(confs), axis=0)[-3:-1, :], axis=0)
    # gm = np.mean(np.stack(confs), axis=0)
    # labels = map(lambda x: x[2][0], netrocs_netaucs_labelsconfs)
    labels = netrocs_netaucs_labelsconfs[0][2][0]

    net_roc = roc_curve(labels, gm)
    net_auc = roc_auc_score(labels, gm)

    plot_roc(
        [('net', net_roc),
         ('prescreener', prescreener_roc)],
        [net_auc, prescreener_auc],
        f'ensemble roc'
    )

    print(time.time() - start)


def test_with_saved():
    root = '/home/maksim/ferit_nets'
    start = time.time()

    tuf_table_file_name = 'fold_0_train.csv'
    train_alarms = tuf_table_csv_to_df(os.path.join(root, 'folds', tuf_table_file_name))
    train_alarms['utm'] = train_alarms['utm'].map(ast.literal_eval)
    tuf_table_file_name = 'fold_0_test.csv'
    test_alarms = tuf_table_csv_to_df(os.path.join(root, 'folds', tuf_table_file_name))
    test_alarms['utm'] = test_alarms['utm'].map(ast.literal_eval)
    all_test_dl = create_dataloader(train_alarms, os.path.join(root, 'data'), batch_size=128)
    region_test_dl = create_dataloader(test_alarms, os.path.join(root, 'data'), batch_size=128,
                                       m=all_test_dl.dataset._mean, s=all_test_dl.dataset._std)

    prescreener_roc = roc_curve(test_alarms['HIT'], test_alarms['conf'])
    prescreener_auc = roc_auc_score(test_alarms['HIT'], test_alarms['conf'])

    def load_net(i):
        net = GPR_15_300()
        net.cuda()
        state_dict = torch.load(f'/home/maksim/ferit_nets/net_fold_{i}.pth')
        new_dict = {}
        for k, v in state_dict.items():
            new_dict[k.replace('module.', '')] = v
        net.load_state_dict(new_dict)
        return net

    nets = map(load_net, range(10))
    netrocs_netaucs_labelsconfs = []
    for i, net in enumerate(nets):
        roc, auc, labels_confs = test(net, region_test_dl)
        netrocs_netaucs_labelsconfs.append((roc, auc, labels_confs))
        plot_roc(
            [('net', roc),
             ('prescreener', prescreener_roc)],
            [auc, prescreener_auc],
            f'holdout roc fold {i}'
        )
    confs = map(lambda x: x[2][1], netrocs_netaucs_labelsconfs)
    gm = np.mean(np.stack(confs), axis=0)
    # labels = map(lambda x: x[2][0], netrocs_netaucs_labelsconfs)
    labels = netrocs_netaucs_labelsconfs[0][2][0]

    net_roc = roc_curve(labels, gm)
    net_auc = roc_auc_score(labels, gm)

    plot_roc(
        [('net', net_roc),
         ('prescreener', prescreener_roc)],
        [net_auc, prescreener_auc],
        f'ensemble roc'
    )

    print(time.time() - start)
    return netrocs_netaucs_labelsconfs, prescreener_roc, prescreener_auc

def prescreener_roc():
    root = '/home/maksim/ferit_nets'
    tuf_table_file_name = 'fold_0_test.csv'
    test_alarms = tuf_table_csv_to_df(os.path.join(root, 'folds', tuf_table_file_name))
    test_alarms['utm'] = test_alarms['utm'].map(ast.literal_eval)
    prescreener_roc = roc_curve(test_alarms['HIT'], test_alarms['conf'])
    prescreener_auc = roc_auc_score(test_alarms['HIT'], test_alarms['conf'])
    plot_roc(
        [
         ('prescreener', prescreener_roc)],
        [prescreener_auc],
        f'F1V4p4v3'
    )

def histograms(confs, labels, title):
    n, bins, patches = plt.hist(np.asarray(confs)[map(bool, labels)], 100, density=False, facecolor='g', alpha=0.75, label='true')
    n, bins, patches = plt.hist(np.asarray(confs)[map(lambda x: not bool(x), labels)], 100, density=False, facecolor='b', alpha=0.75, label='false')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def test_with_cross_val(alarms):
    root = '/home/maksim/ferit_nets'
    nets = defaultdict(list)
    for i, (train_splits, region_train_data, region_test_data) in enumerate(
            region_and_stratified(alarms, n_splits_region=3, n_splits_stratified=10)):

        for j, (train_data, validation_data) in enumerate(train_splits):
            print(len(train_data), len(validation_data), len(region_train_data), len(region_test_data))
            train_dl = create_dataloader(train_data, os.path.join(root, 'data'), batch_size=128)
            validation_set_dl = create_dataloader(validation_data, os.path.join(root, 'data'), batch_size=128,
                                                  m=train_dl.dataset._mean, s=train_dl.dataset._std)
            prescreener_roc = roc_curve(validation_data['HIT'], validation_data['conf'])
            prescreener_auc = roc_auc_score(validation_data['HIT'], validation_data['conf'])

            net_name = f'"act" {i} fold {j}'

            print(net_name)
            net = train(train_dl, 0)
            net_roc, net_auc, _ = test(net, validation_set_dl)
            plot_roc(
                [('net', net_roc), ('prescreener', prescreener_roc)],
                [net_auc, prescreener_auc],
                f'before train roc act {i} fold {j}'
            )

            net = train(train_dl, 1)
            net_roc, net_auc, _ = test(net, validation_set_dl)
            if .49 < net_auc < .51:
                continue
            plot_roc(
                [('net', net_roc), ('prescreener', prescreener_roc)],
                [net_auc, prescreener_auc],
                f'after train roc act {i} fold {j}'
            )
            nets[f'"act {i}'].append(net)

        all_test_dl = create_dataloader(region_train_data, os.path.join(root, 'data'), batch_size=128)
        region_test_dl = create_dataloader(region_test_data, os.path.join(root, 'data'), batch_size=128,
                                           m=all_test_dl.dataset._mean, s=all_test_dl.dataset._std)

        prescreener_roc = roc_curve(region_test_data['HIT'], region_test_data['conf'])
        prescreener_auc = roc_auc_score(region_test_data['HIT'], region_test_data['conf'])
        netrocs_netaucs_labelsconfs = map(lambda net: test(net, region_test_dl), nets[f'"act {i}'])
        confs = map(lambda x: x[2][1], netrocs_netaucs_labelsconfs)
        gm = np.mean(np.stack(confs), axis=0)
        # labels = map(lambda x: x[2][0], netrocs_netaucs_labelsconfs)
        labels = netrocs_netaucs_labelsconfs[0][2][0]

        net_roc = roc_curve(labels, gm)
        net_auc = roc_auc_score(labels, gm)

        plot_roc(
            [('net', net_roc),
             ('prescreener', prescreener_roc)],
            [net_auc, prescreener_auc],
            f'ensemble roc act {i}'
        )
        # plot_roc(
        #     [*map(lambda x: (f'net fold {x[0]}', x[1][0]), enumerate(netrocs_netaucs_labelsconfs)),
        #      ('prescreener', prescreener_roc)],
        #     [*map(itemgetter(1), netrocs_netaucs_labelsconfs), prescreener_auc],
        #     f'after train roc act {i}'
        # )


def plot_hists_saved():
    res = test_with_saved()
    confs = map(lambda x: x[2][1], res[0])
    labels = map(lambda x: x[2][0], res[0])
    for i in range(len(confs)):
        histograms(confs[i], labels[i], f'net {i}')


if __name__ == '__main__':
    root = os.getcwd()
    tuf_table_file_name = 'all_maxs.csv'
    # all_alarms = tuf_table_csv_to_df(os.path.join(root, tuf_table_file_name))
    # test_with_cross_val(all_alarms)
    # test_fold_0()
    # prescreener_roc()
    plot_hists_saved()
