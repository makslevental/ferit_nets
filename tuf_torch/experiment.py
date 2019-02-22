import datetime
import os
import random
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats.mstats import gmean
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from tuf_torch.config import (
    DATA_ROOT, PROJECT_ROOT,
    BATCH_SIZE,
    SHUFFLE_DL,
    OPTIMIZER,
    SCHEDULER,
    CRITERION,
    EPOCHS,
    N_STRAT_SPLITS
)
from tuf_torch.cross_val import tuf_table_csv_to_df, region_and_stratified
from tuf_torch.data import AlarmDataset, Normalize
from tuf_torch.models427 import GPR_15_300
from tuf_torch.test import test
from tuf_torch.train import train
from tuf_torch.visualization import plot_roc

tuf_table_file_name = 'all_maxs.csv'
all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, tuf_table_file_name))


def deriv(x, y):
    if len(x) == 1:
        return y, y
    dy = np.diff(y, 1)
    dx = np.diff(x, 1)
    yfirst = dy / dx

    xfirst = 0.5 * (x[:-1] + x[1:])
    dyfirst = np.diff(yfirst, 1)
    dxfirst = np.diff(xfirst, 1)
    ysecond = dyfirst / dxfirst
    return yfirst, ysecond


word_file = "/usr/share/dict/words"
WORDS = open(word_file).read().splitlines()

PROJECT_NAME = random.choice(WORDS)
print(PROJECT_NAME)

NETS_PATH = os.path.join(PROJECT_ROOT, "nets", f"{PROJECT_NAME}_{datetime.date.today()}")
FIGS_PATH = os.path.join(PROJECT_ROOT, "figs", f"{PROJECT_NAME}_{datetime.date.today()}")
if not os.path.exists(NETS_PATH):
    os.makedirs(NETS_PATH)
if not os.path.exists(FIGS_PATH):
    os.makedirs(FIGS_PATH)

m1, s1 = torch.serialization.load(os.path.join(PROJECT_ROOT, 'means.pt')), \
         torch.serialization.load(os.path.join(PROJECT_ROOT, 'stds.pt'))

for i, (strat_splits, _rgn_train, rgn_holdout) in enumerate(
        region_and_stratified(all_alarms, n_splits_stratified=N_STRAT_SPLITS)):
    fig_train = None
    fig_test = None
    nets = []

    # main training loop
    for j, (alarm_strat_train, alarm_strat_holdout) in enumerate(strat_splits):
        # train
        strat_train_ad = AlarmDataset(
            alarm_strat_train,
            DATA_ROOT,
            transform=transforms.Compose([Normalize(m1, s1)])
        )
        strat_train_adl = DataLoader(strat_train_ad, BATCH_SIZE, SHUFFLE_DL, num_workers=4)
        net = torch.nn.DataParallel(GPR_15_300())
        optim = OPTIMIZER(net)
        sched = SCHEDULER(optim, strat_train_adl)
        candidate_net_aucs = []
        for candidate_net in train(
                net,
                strat_train_adl,
                criterion=CRITERION,
                optimizer=optim,
                scheduler=sched,
                epochs=EPOCHS
        ):
            # test for early stopping using flattening of auc curve
            holdout_ad = AlarmDataset(
                alarm_strat_holdout,
                DATA_ROOT,
                transform=transforms.Compose([Normalize(m1, s1)])
            )
            holdout_adl = DataLoader(holdout_ad, BATCH_SIZE, SHUFFLE_DL, num_workers=4)
            _roc, auc, _all_labels, _confs = test(net, holdout_adl)
            candidate_net_aucs.append(auc)
            aucp, aucpp = deriv(np.arange(len(candidate_net_aucs)), candidate_net_aucs)
            if len(candidate_net_aucs) >= 3 and all(aucp[-2:] >= 0) and all(aucpp[-2:0] <= 0):
                break

        nets.append((auc, candidate_net))
        print(f"done with {i} {j} train")

    # drop worst (by auc) 2 nets
    nets = list(map(itemgetter(1), sorted(nets, key=itemgetter(0))[2:]))
    # majority = len(nets) // 2 + 1
    rgn_holdout_ad = AlarmDataset(
        rgn_holdout,
        DATA_ROOT,
        transform=transforms.Compose([Normalize(m1, s1)])
    )

    ####
    # testing on holdoout
    ####

    # DO NOT SHUFFLE region holdout in order to fuse
    rgn_holdout_adl = DataLoader(rgn_holdout_ad, BATCH_SIZE, shuffle=False, num_workers=4)
    for j, net in enumerate(nets):
        torch.save(net.state_dict(), os.path.join(NETS_PATH, f"net_test_{i}_{j}.net"))
        roc, auc, labels, confs = test(net, rgn_holdout_adl)
        fig_test = plot_roc(roc, f"test {i}", f"{auc}", show=False, fig=fig_test)

        all_aucs = all_aucs + [auc] if j > 0 else [auc]
        all_confs = np.vstack((all_confs, confs)) if j > 0 else confs
        all_labels = np.vstack((all_labels, labels)) if j > 0 else labels
    fig_test.savefig(os.path.join(FIGS_PATH, f'test_{i}.png'))

    # fusion
    fused_confs = gmean(all_confs, axis=0)
    fused_roc = roc_curve(all_labels[0, :], fused_confs)
    if len(set(all_labels[0, :])) > 1:
        fused_auc = roc_auc_score(all_labels[0, :], fused_confs)
    else:
        fused_auc = 'NaN'
    fig_fused = plot_roc(fused_roc, f"fused {i} roc", f"auc {fused_auc}")
    fig_fused.savefig(os.path.join(FIGS_PATH, f'fused_{i}.png'))

    # histograms
    hist_range = (np.min(all_confs), np.max(all_confs))
    hist_fig = plt.figure()
    ax = hist_fig.add_axes([0.1, 0.1, 0.85, 0.8])
    ax.set_title(f"hist {i}")
    for j, confs in enumerate(all_confs):
        ax.hist(confs, 100, range=hist_range, density=True, label=all_aucs[j], alpha=0.7)
    hist_fig.legend(loc='upper right')
    hist_fig.savefig(os.path.join(FIGS_PATH, f'hist_{i}.png'))

    plt.close(fig_test)
    plt.close(fig_fused)
    plt.close(hist_fig)
