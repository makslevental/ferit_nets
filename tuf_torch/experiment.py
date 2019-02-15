import os

import numpy as np
import torch
from scipy.stats.mstats import gmean, hmean, hdmedian
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
from tuf_torch.data import AlarmDataset
from tuf_torch.models427 import GPR_15_300
from tuf_torch.test import test
from tuf_torch.train import train
from tuf_torch.visualization import plot_roc

tuf_table_file_name = 'big_maxs_table.csv'
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


fig_test = None
for i, (strat_splits, rgn_train, rgn_test) in enumerate(
        region_and_stratified(all_alarms, n_splits_stratified=N_STRAT_SPLITS)):
    fig_train = None
    nets = []

    for j, (noholdout, holdout) in enumerate(strat_splits):
        # train
        notholdout_ad = AlarmDataset(
            noholdout,
            DATA_ROOT,
        )
        noholdout_adl = DataLoader(notholdout_ad, BATCH_SIZE, SHUFFLE_DL, num_workers=4)
        net = torch.nn.DataParallel(GPR_15_300())
        optim = OPTIMIZER(net)
        sched = SCHEDULER(optim, noholdout_adl)
        aucs = []
        for candidate_net in train(
                net,
                noholdout_adl,
                criterion=CRITERION,
                optimizer=optim,
                scheduler=sched,
                epochs=EPOCHS
        ):
            # test
            holdout_ad = AlarmDataset(
                holdout,
                DATA_ROOT,
            )
            holdout_adl = DataLoader(holdout_ad, BATCH_SIZE, SHUFFLE_DL, num_workers=4)
            roc, auc, all_labels, confs = test(net, holdout_adl)
            aucs.append(auc)
            aucp, aucpp = deriv(np.arange(len(aucs)), aucs)
            if len(aucs) >= 3 and all(aucp[-2:] >= 0) and all(aucpp[-2:0] <= 0):
                break

        fig_train = plot_roc(
            roc,
            'train',
            f'{i} {j} auc {auc:.3f}',
            fig=fig_train
        )
        print(f"done with {i} {j} train")
        nets.append(candidate_net)

    fig_train.show()

    majority = len(nets) // 2 + 1
    rgn_test_ad = AlarmDataset(
        rgn_test,
        DATA_ROOT,
    )
    rgn_test_adl = DataLoader(rgn_test_ad, BATCH_SIZE, SHUFFLE_DL, num_workers=4)
    _roc, _auc, _all_labels, all_confs = test(nets[0], rgn_test_adl)
    for j, net in enumerate(nets[1:]):
        _roc, auc, all_labels, confs = test(net, rgn_test_adl)
        all_confs = np.vstack((all_confs, confs))

    sorted_confs = np.sort(all_confs, axis=0)
    oracle_roc = []
    for i, label in enumerate(all_labels):
        if label == 1:
            oracle_roc.append(sorted_confs[-1, i])
        else:
            oracle_roc.append(sorted_confs[0, i])

    rocs = [
        roc_curve(all_labels, np.sort(all_confs, axis=0)[6, :]),
        roc_curve(all_labels, oracle_roc),
        roc_curve(all_labels, np.mean(all_confs, axis=0)),
        roc_curve(all_labels, gmean(all_confs, axis=0)),
        roc_curve(all_labels, hmean(all_confs, axis=0)),
        roc_curve(all_labels, hdmedian(all_confs, axis=0)),
    ]

    for roc in rocs:
        fig_test = plot_roc(
            roc,
            f'test',
            f'{i}',
            fig=fig_test
        )
    hist_range = (np.min(all_confs), np.max(all_confs))
    hist_fig = plt.figure()
    ax = hist_fig.add_axes([0.1, 0.1, 0.85, 0.8])
    for i, confs in enumerate(all_confs):
        ax.hist(confs, 100, range=hist_range, density=True, label=f'{i}', alpha=0.5)
    hist_fig.legend(loc='upper right')
    hist_fig.show()
    print(f"done with {i} test")

fig_test.show()
