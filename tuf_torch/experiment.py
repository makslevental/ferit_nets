import os
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from defaultlist import defaultlist
from scipy.stats.mstats import gmean
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from tuf_torch.config import (
    DATA_ROOT, PROJECT_ROOT, PROJECT_NAME,
    BATCH_SIZE,
    SHUFFLE_DL,
    OPTIMIZER,
    SCHEDULER,
    CRITERION,
    EPOCHS,
    N_STRAT_SPLITS,
    LOGS_PATH, NETS_PATH, FIGS_PATH)
from tuf_torch.cross_val import tuf_table_csv_to_df, region_and_stratified
from tuf_torch.data import AlarmDataset, Normalize, load_nets_dir
from tuf_torch.models427 import GPR_15_300, AucLoss
from tuf_torch.test import test
from tuf_torch.train import train
from tuf_torch.visualization import plot_roc, plot_auc_loss


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


def experiment1():
    tuf_table_file_name = 'all_maxs.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, tuf_table_file_name))

    print(PROJECT_NAME)

    m1, s1 = torch.serialization.load(os.path.join(PROJECT_ROOT, 'means.pt')), \
             torch.serialization.load(os.path.join(PROJECT_ROOT, 'stds.pt'))

    for i, (strat_splits, _rgn_train, rgn_holdout) in enumerate(
            region_and_stratified(all_alarms, n_splits_stratified=N_STRAT_SPLITS)):
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

        fig_test = None
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


def exp2():
    tuf_table_file_name = 'all_maxs.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, tuf_table_file_name))

    m1, s1 = torch.serialization.load(os.path.join(PROJECT_ROOT, 'means.pt')), \
             torch.serialization.load(os.path.join(PROJECT_ROOT, 'stds.pt'))
    nets = load_nets_dir(NETS_PATH, "GPR_15_300")

    for i, (strat_splits, _rgn_train, rgn_holdout) in enumerate(
            region_and_stratified(all_alarms, n_splits_stratified=N_STRAT_SPLITS)):
        rgn_holdout_ad = AlarmDataset(
            rgn_holdout,
            DATA_ROOT,
            transform=transforms.Compose([Normalize(m1, s1)])
        )
        fig_test = None
        # DO NOT SHUFFLE region holdout in order to fuse
        rgn_holdout_adl = DataLoader(rgn_holdout_ad, BATCH_SIZE, shuffle=False, num_workers=4)
        for j, net in enumerate(nets):
            # torch.save(net.state_dict(), os.path.join(NETS_PATH, f"net_test_{i}_{j}.net"))
            roc, auc, labels, confs = test(net, rgn_holdout_adl)
            fig_test = plot_roc(roc, f"test {i}", f"{auc}", show=False, fig=fig_test)

            all_aucs = all_aucs + [auc] if j > 0 else [auc]
            all_confs = np.vstack((all_confs, confs)) if j > 0 else confs
            all_labels = np.vstack((all_labels, labels)) if j > 0 else labels
            print(f"done testing {i} {j}")
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


def exp3():
    tuf_table_file_name = 'all_maxs.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, "csvs", tuf_table_file_name))
    australia_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, "csvs", "australia.csv"))

    m1, s1 = torch.serialization.load(os.path.join(PROJECT_ROOT, 'means.pt')), \
             torch.serialization.load(os.path.join(PROJECT_ROOT, 'stds.pt'))

    australia_ad = AlarmDataset(
        australia_alarms,
        DATA_ROOT,
        transform=transforms.Compose([Normalize(m1, s1)])
    )
    australia_adl = DataLoader(australia_ad, BATCH_SIZE, shuffle=False, num_workers=4)

    f = open(os.path.join(LOGS_PATH, "loss.csv"), "w+")

    nets = defaultlist(lambda: torch.nn.DataParallel(GPR_15_300()))
    for overtrain in range(100):
        for i, (strat_splits, _rgn_train, rgn_holdout) in enumerate(
                region_and_stratified(all_alarms, n_splits_stratified=N_STRAT_SPLITS)):

            net = nets[i]
            # main training loop
            for j, (alarm_strat_train, alarm_strat_holdout) in enumerate(strat_splits):
                # train
                strat_train_ad = AlarmDataset(
                    alarm_strat_train,
                    DATA_ROOT,
                    transform=transforms.Compose([Normalize(m1, s1)])
                )
                strat_train_adl = DataLoader(strat_train_ad, BATCH_SIZE, SHUFFLE_DL, num_workers=4)

                optim = OPTIMIZER(net)
                sched = SCHEDULER(optim, strat_train_adl)
                for k, _ in enumerate(train(
                        net,
                        strat_train_adl,
                        criterion=CRITERION,
                        optimizer=optim,
                        scheduler=sched,
                        epochs=EPOCHS
                )):
                    _roc, auc, _all_labels, _confs, loss = test(net, australia_adl, CRITERION)

                    f.write(f"{i}, {j}, {k}, {auc}, {loss}\n")
                    f.flush()
                    print(f"done with {i} {j} {k}  auc: {auc} loss: {loss}")


def exp4():
    tuf_table_file_name = 'all_maxs.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, "csvs", tuf_table_file_name))
    australia_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, "csvs", "australia.csv"))

    m1, s1 = torch.serialization.load(os.path.join(PROJECT_ROOT, 'means.pt')), \
             torch.serialization.load(os.path.join(PROJECT_ROOT, 'stds.pt'))

    australia_ad = AlarmDataset(
        australia_alarms,
        DATA_ROOT,
        transform=transforms.Compose([Normalize(m1, s1)])
    )
    australia_adl = DataLoader(australia_ad, BATCH_SIZE, shuffle=False, num_workers=4)

    f = open(os.path.join(LOGS_PATH, "loss.csv"), "w+")

    for i, (strat_splits, _rgn_train, rgn_holdout) in enumerate(
            region_and_stratified(all_alarms, n_splits_stratified=N_STRAT_SPLITS)):
        rgn_holdout_ad = AlarmDataset(
            rgn_holdout,
            DATA_ROOT,
            transform=transforms.Compose([Normalize(m1, s1)])
        )
        rgn_holdout_adl = DataLoader(rgn_holdout_ad, BATCH_SIZE, shuffle=False, num_workers=4)
        break

    aus_aucs, rgn_aucs = [0, 0, 0], [0, 0, 0]
    net = torch.nn.DataParallel(GPR_15_300())
    for overtrain in range(100):
        # main training loop
        for j, (alarm_strat_train, alarm_strat_holdout) in enumerate(strat_splits):
            # train
            strat_train_ad = AlarmDataset(
                alarm_strat_train,
                DATA_ROOT,
                transform=transforms.Compose([Normalize(m1, s1)])
            )
            strat_train_adl = DataLoader(strat_train_ad, BATCH_SIZE, shuffle=True, num_workers=4)
            strat_holdout_ad = AlarmDataset(
                alarm_strat_holdout,
                DATA_ROOT,
                transform=transforms.Compose([Normalize(m1, s1)])
            )
            strat_holdout_adl = DataLoader(strat_holdout_ad, BATCH_SIZE, shuffle=True, num_workers=4)

            optim = OPTIMIZER(net)
            sched = SCHEDULER(optim, strat_train_adl)
            for k, _ in enumerate(train(
                    net,
                    strat_train_adl,
                    criterion=CRITERION,
                    optimizer=optim,
                    scheduler=sched,
                    epochs=EPOCHS
            )):
                _roc, aus_auc, _all_labels, aus_confs, aus_loss = test(net, australia_adl, CRITERION)
                _roc, rgn_auc, _all_labels, rgn_confs, rgn_loss = test(net, rgn_holdout_adl, CRITERION)
                _roc, strat_auc, _all_labels, strat_confs, strat_loss = test(net, strat_holdout_adl, CRITERION)

                f.write(
                    f"{overtrain}, {i}, {j}, {k}, {aus_auc}, {aus_loss}, {rgn_auc}, {rgn_loss}, {strat_auc}, {strat_loss}\n")
                f.flush()
                print(
                    f"{overtrain}, {i}, {j}, {k}, {aus_auc}, {aus_loss}, {rgn_auc}, {rgn_loss}, {strat_auc}, {strat_loss}\n")
                aus_aucs[k % 3] = aus_auc
                rgn_aucs[k % 3] = rgn_auc
                if np.abs(np.mean(aus_aucs) - 0.5) < 0.0000001 or np.abs(np.mean(rgn_aucs) - 0.5) < 0.00000001:
                    print(f"!!!!diverged {np.mean(aus_confs)} {np.mean(rgn_confs)}")
                    net = torch.nn.DataParallel(GPR_15_300())
                    break


def exp5():
    tuf_table_file_name = 'all_maxs.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, "csvs", tuf_table_file_name))
    m1, s1 = torch.serialization.load(os.path.join(PROJECT_ROOT, 'means.pt')), \
             torch.serialization.load(os.path.join(PROJECT_ROOT, 'stds.pt'))

    log_csv = open(os.path.join(LOGS_PATH, "loss.csv"), "w+")
    log_csv.write("overtrain_epoch, net, train_epoch, strat_auc, strat_loss\n")

    for i, (strat_splits, _rgn_train, rgn_holdout) in enumerate(
            region_and_stratified(all_alarms, n_splits_stratified=N_STRAT_SPLITS)):
        break

    net = torch.nn.DataParallel(GPR_15_300())
    strat_aucs = [0, 0, 0]
    strat_train_ad = None
    strat_train_adl = None
    strat_holdout_ad = None
    strat_holdout_adl = None
    for i in range(100):
        # main training loop
        for j, (alarm_strat_train, alarm_strat_holdout) in enumerate(strat_splits):
            # train
            strat_train_ad = strat_train_ad if strat_train_ad is not None else AlarmDataset(
                alarm_strat_train,
                DATA_ROOT,
                transform=transforms.Compose([Normalize(m1, s1)])
            )
            strat_train_adl = strat_train_adl if strat_train_adl is not None else DataLoader(strat_train_ad, BATCH_SIZE,
                                                                                             shuffle=True,
                                                                                             num_workers=4)

            strat_holdout_ad = strat_holdout_ad if strat_holdout_ad is not None else AlarmDataset(
                alarm_strat_holdout,
                DATA_ROOT,
                transform=transforms.Compose([Normalize(m1, s1)])
            )
            strat_holdout_adl = strat_holdout_adl if strat_holdout_adl is not None else DataLoader(strat_holdout_ad,
                                                                                                   BATCH_SIZE,
                                                                                                   shuffle=True,
                                                                                                   num_workers=4)

            optim = OPTIMIZER(net)
            sched = SCHEDULER(optim, strat_train_adl)
            for k, _ in enumerate(train(
                    net,
                    strat_train_adl,
                    criterion=CRITERION,
                    optimizer=optim,
                    scheduler=sched,
                    epochs=EPOCHS
            )):
                _roc, strat_auc, _all_labels, strat_confs, strat_loss = test(net, strat_holdout_adl, CRITERION)

                log_csv.write(
                    f"{i}, {j}, {k}, {strat_auc}, {strat_loss}\n")
                log_csv.flush()
                print(
                    f"{i}, {j}, {k}, {strat_auc}, {strat_loss}\n")
                strat_aucs[k % 3] = strat_auc
                if np.abs(np.mean(strat_aucs) - 0.5) < 0.0000001 or np.abs(np.mean(strat_aucs) - 0.5) < 0.00000001:
                    print(f"!!!! diverged")
                    net = torch.nn.DataParallel(GPR_15_300())
                    break

            break


def exp6():
    tuf_table_file_name = 'big_maxs_table.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, "csvs", tuf_table_file_name))
    m1, s1 = torch.serialization.load(os.path.join(PROJECT_ROOT, 'means.pt')), \
             torch.serialization.load(os.path.join(PROJECT_ROOT, 'stds.pt'))

    log_path = os.path.join(LOGS_PATH, "loss.csv")
    log_csv = open(log_path, "w")
    log_csv.write("overtrain_epoch, net, train_epoch, strat_auc, strat_loss\n")

    for i, (strat_splits, _rgn_train, rgn_holdout) in enumerate(
            region_and_stratified(all_alarms, n_splits_stratified=N_STRAT_SPLITS)):
        break

    net = torch.nn.DataParallel(GPR_15_300())
    strat_aucs = [0, 0, 0]
    strat_train_ad = None
    strat_train_adl = None
    strat_holdout_ad = None
    strat_holdout_adl = None
    for i in range(100):
        # main training loop
        for j, (alarm_strat_train, alarm_strat_holdout) in enumerate(strat_splits):
            # train
            strat_train_ad = strat_train_ad if strat_train_ad is not None else AlarmDataset(
                alarm_strat_train,
                DATA_ROOT,
                transform=transforms.Compose([Normalize(m1, s1)])
            )
            strat_train_adl = strat_train_adl if strat_train_adl is not None else DataLoader(strat_train_ad, BATCH_SIZE,
                                                                                             shuffle=True,
                                                                                             num_workers=4)

            strat_holdout_ad = strat_holdout_ad if strat_holdout_ad is not None else AlarmDataset(
                alarm_strat_holdout,
                DATA_ROOT,
                transform=transforms.Compose([Normalize(m1, s1)])
            )
            strat_holdout_adl = strat_holdout_adl if strat_holdout_adl is not None else DataLoader(strat_holdout_ad,
                                                                                                   BATCH_SIZE,
                                                                                                   shuffle=True,
                                                                                                   num_workers=4)

            optim = OPTIMIZER(net)
            sched = SCHEDULER(optim, strat_train_adl)
            for k, _ in enumerate(train(
                    net,
                    strat_train_adl,
                    # criterion=AucLoss(),
                    criterion=CRITERION,
                    optimizer=optim,
                    scheduler=sched,
                    epochs=EPOCHS
            )):
                _roc, strat_auc, _all_labels, strat_confs, strat_loss = test(net, strat_holdout_adl, CRITERION)

                log_csv.write(
                    f"{i}, {j}, {k}, {strat_auc}, {strat_loss}\n")
                log_csv.flush()
                print(
                    f"{i}, {j}, {k}, {strat_auc}, {strat_loss}\n")
                strat_aucs[k % 3] = strat_auc
                if np.abs(np.mean(strat_aucs) - 0.5) < 0.0000001 or np.abs(np.mean(strat_aucs) - 0.5) < 0.00000001:
                    print(f"!!!! diverged")
                    net = torch.nn.DataParallel(GPR_15_300())
                    break

            break

    log_csv.close()
    log_df = pd.read_csv(log_path, sep=",\s+")
    plot_auc_loss(log_df, "cross entropy loss")


def exp7():
    tuf_table_file_name = 'big_maxs_table.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(PROJECT_ROOT, "csvs", tuf_table_file_name))
    m1, s1 = torch.serialization.load(os.path.join(PROJECT_ROOT, 'means.pt')), \
             torch.serialization.load(os.path.join(PROJECT_ROOT, 'stds.pt'))
    nets = defaultlist(lambda: defaultlist(lambda: torch.nn.DataParallel(GPR_15_300())))
    for i, (strat_splits, _rgn_train, rgn_holdout) in enumerate(
            region_and_stratified(all_alarms, n_splits_stratified=N_STRAT_SPLITS)):
        # main training loop
        for j, (alarm_strat_train, alarm_strat_holdout) in enumerate(strat_splits):
            net = nets[i][j]
            strat_train_ad = AlarmDataset(
                alarm_strat_train,
                DATA_ROOT,
                transform=transforms.Compose([Normalize(m1, s1)])
            )
            strat_train_adl = DataLoader(strat_train_ad, BATCH_SIZE, SHUFFLE_DL, num_workers=16)
            optim = OPTIMIZER(net)
            sched = SCHEDULER(optim, strat_train_adl)
            for k, (_, loss) in enumerate(train(
                    net,
                    strat_train_adl,
                    criterion=AucLoss(),
                    optimizer=optim,
                    scheduler=sched,
                    epochs=EPOCHS
            )):
                print(i, j, k, loss)

    for i, sub_nets in enumerate(nets):
        for j, net in enumerate(sub_nets):
            torch.save(net.state_dict(), os.path.join(NETS_PATH, f"net_test_{i}_{j}.net"))


if __name__ == "__main__":
    exp7()
