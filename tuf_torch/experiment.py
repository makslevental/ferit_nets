import os

import torch
from torch.utils.data import DataLoader

from tuf_torch.config import DATA_ROOT, PROJECT_ROOT, BATCH_SIZE, SHUFFLE_DL, OPTIMIZER, SCHEDULER, CRITERION, EPOCHS, \
    N_STRAT_SPLITS
from tuf_torch.cross_val import tuf_table_csv_to_df, region_and_stratified
from tuf_torch.data import AlarmDataset
from tuf_torch.models427 import GPR_15_300
from tuf_torch.test import test
from tuf_torch.train import train
from tuf_torch.visualization import plot_roc

tuf_table_file_name = 'big_maxs_table.csv'
all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, tuf_table_file_name))
net = torch.nn.DataParallel(GPR_15_300())

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
        optim = OPTIMIZER(net)
        sched = SCHEDULER(optim, noholdout_adl)
        net = train(
            net,
            noholdout_adl,
            criterion=CRITERION,
            optimizer=optim,
            scheduler=sched,
            epochs=EPOCHS
        )

        # test
        holdout_ad = AlarmDataset(
            holdout,
            DATA_ROOT,
        )
        holdout_adl = DataLoader(holdout_ad, BATCH_SIZE, SHUFFLE_DL, num_workers=4)
        roc, auc, all_labels, confs = test(net, holdout_adl)
        fig_train = plot_roc(
            roc,
            'train',
            f'{i} {j} auc {auc:.3f}',
            fig=fig_train
        )
        nets.append(net)
        print(f"done with {i} {j} train")

    fig_train.show()

    rgn_test_ad = AlarmDataset(
        rgn_test,
        DATA_ROOT,
    )
    fig_test = None
    rgn_test_adl = DataLoader(rgn_test_ad, BATCH_SIZE, SHUFFLE_DL, num_workers=4)
    for j, net in enumerate(nets):
        roc, auc, all_labels, confs = test(net, rgn_test_adl)
        fig_test = plot_roc(
            roc,
            f' test',
            f'{i} {j} auc {auc:.3f}',
            fig=fig_test
        )
        print(f"done with {i} {j} test")
    fig_test.show()
