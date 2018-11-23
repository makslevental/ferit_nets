import os
from collections import defaultdict
from operator import itemgetter
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import (StratifiedKFold, KFold, GroupKFold, BaseCrossValidator, GroupShuffleSplit,
                                     StratifiedShuffleSplit, ShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold)
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from visualization import plot_cv_indices, visualize_groups

DEBUG = False

from util import *


class AlarmDataset(Dataset):
    def __init__(self, csv_file, root_dir, feature_type='NNsig_NoInt_15_300', transform=None):
        self.alarms_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.feature_type = feature_type

    def __len__(self):
        return len(self.alarms_frame)

    def __getitem__(self, idx):
        row = self.alarms_frame.iloc[idx]
        file_name = os.path.join(
            self.root_dir,
            row['sample'],
            row['event_id']
        )
        file = h5py.File(f'{file_name}.h5', 'r')

        feature = np.array(file[self.feature_type])
        hit = row['HIT']
        if self.transform:
            feature = self.transform(feature)

        return feature, hit


def tuf_table_csv_to_df(fp) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df = df.where((pd.notnull(df)), None)

    def corners_to_tuple(corners: pd.DataFrame) -> Optional[List[Tuple]]:
        corners = filter(None, corners)
        num_corners = len(corners)
        corner_tuples = zip(corners[:num_corners // 2], corners[num_corners // 2:])
        return corner_tuples if len(corner_tuples) else None

    corners_names = filter(lambda x: 'corner' in x, df.columns)
    df['corners'] = df[corners_names].apply(corners_to_tuple, axis=1)
    df.drop(corners_names, inplace=True, axis=1)

    df.drop(
        [
            'xUTM', 'yUTM', 'sensor_type', 'dt', 'ch', 'scan', 'time', 'MineIsInsideLane', 'DIST', 'objectdist',
            'category', 'isDetected', 'IsInsideLane', 'prescreener', 'known_obj', 'event_type', 'id', 'lsd_kv',
            'encoder', 'event_timestamp', 'fold'
        ],
        inplace=True,
        axis=1
    )

    return df


def group_alarms_by(
        df: pd.DataFrame,
        attrs: List[str] = None,
        group_attr: List[str] = None
) -> Tuple[List[Tuple[int, pd.DataFrame]], List[Tuple[int, int]]]:
    if attrs is None: attrs = ['srid', 'depth', 'corners']

    if DEBUG:
        # intellij doesn't display dataframes with tuple entries for some reason
        # so do this for debug purposes
        def get_group(df, idx, attrs):
            row = df.loc[idx][attrs]
            # corners are a list and lists can't be hashed
            if 'corners' in attrs:
                row['corners'] = tuple(row['corners'])
            return tuple(row)

        gb = df.dropna(subset=['corners']).groupby(lambda idx: get_group(df, idx, attrs), sort=False)
    else:
        if 'corners' in attrs:
            df['corners'] = df['corners'].map(tuple, na_action='ignore')

        gb = df.dropna(subset=['corners']).groupby(attrs, sort=False)

    groups = list(gb)

    if group_attr:
        group_df_map = defaultdict(list)
        for _, group in groups:
            group_key = tuple(group.iloc[0][group_attr])
            group_df_map[group_key].append(group)
        groups_dfs = list(enumerate(group_df_map.values()))
    else:
        groups_dfs = [(group, [df]) for group, (_, df) in enumerate(groups)]

    misses = df[df['HIT'] == 0]
    groups_dfs.append((len(groups_dfs), np.split(misses, len(misses))))
    dfs_groups = []
    for group, dfs in groups_dfs:
        for df in dfs:
            dfs_groups.extend([(idx, group) for idx in df.index])
    return groups_dfs, sorted(dfs_groups, key=itemgetter(0))


def create_cross_val_splits(
        n_splits: int,
        cv: BaseCrossValidator,
        all_alarms: pd.DataFrame,
        grouped_alarms: List[Tuple[int, pd.DataFrame]],
        groups=None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    # if number of instances/grouping is less than the number of folds then you can't distribute across the folds evenly

    enough_groups = filter(lambda g_df: len(g_df[1]) >= n_splits, grouped_alarms)

    # if len(enough_groups):
    #     print(grouped_alarms)
    enough_group_xs = np.hstack(map(lambda g_dfs: np.hstack(map(lambda df: df.index, g_dfs[1])), enough_groups))
    enough_group_labels = np.hstack(map(lambda g_df: [g_df[0]] * sum(map(len, g_df[1])), enough_groups))

    not_enough_groups = filter(lambda g_df: len(g_df[1]) < n_splits, grouped_alarms)
    not_enough_group_idxs = np.hstack(map(lambda g_dfs: np.hstack(map(lambda df: df.index, g_dfs[1])), not_enough_groups))

    assert not set(enough_group_xs) & set(not_enough_group_idxs)

    if getattr(cv, 'shuffle', None):
        # in place
        np.random.shuffle(not_enough_group_idxs)

    not_enough_group_splits = np.array_split(
        not_enough_group_idxs,
        n_splits
    )

    alarm_splits = []
    cv_iter = cv.split(
        X=enough_group_xs,
        y=enough_group_labels,
        groups=groups or enough_group_labels
    )
    # these are indices in enough_group_xs not in the dataframe (because that's how sklearn works)
    for i, (enough_group_train_idxs, enough_group_test_idxs) in enumerate(cv_iter):
        # these are now indices in the dataframe
        enough_group_train_df_idxs, enough_group_test_df_idxs = enough_group_xs[enough_group_train_idxs], \
                                                                enough_group_xs[enough_group_test_idxs]

        j = i % n_splits
        not_enough_group_train_idxs = np.hstack(not_enough_group_splits[:j] + not_enough_group_splits[j + 1:])
        not_enough_group_test_idxs = not_enough_group_splits[j]

        assert not set(enough_group_train_df_idxs) & set(enough_group_test_df_idxs)
        assert not set(not_enough_group_train_idxs) & set(not_enough_group_test_idxs)

        train_idxs = np.hstack([enough_group_train_df_idxs, not_enough_group_train_idxs])
        test_idxs = np.hstack([enough_group_test_df_idxs, not_enough_group_test_idxs])

        assert not set(train_idxs) & set(test_idxs)

        alarm_splits.append(
            (all_alarms.loc[train_idxs],
             all_alarms.loc[test_idxs])
        )

    return alarm_splits


def visualize_cross_vals(df: pd.DataFrame, n_splits=10):
    groups_dfs, dfs_groups = group_alarms_by(df)
    visualize_groups(groups_dfs, dfs_groups)

    cvs = [
        KFold(n_splits=n_splits),
        GroupKFold(n_splits=n_splits),
        ShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=0),
        StratifiedKFold(n_splits=n_splits, shuffle=False),
        GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0),
        StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0),
        RepeatedKFold(n_splits=n_splits, n_repeats=2, random_state=0),
        RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=2, random_state=0),
    ]

    for cv in cvs:
        try:
            cross_val_splits = create_cross_val_splits(n_splits, cv, df, groups_dfs)
            plot_cv_indices(cv, cross_val_splits, dfs_groups)
        except Exception as e:
            print(e, type(cv).__name__)


root = os.getcwd()
# tuf_table_file_name = 'three_maxs_table.csv'
tuf_table_file_name = 'all_maxs.csv'
df = tuf_table_csv_to_df(os.path.join(root, tuf_table_file_name))
visualize_cross_vals(df, n_splits=10)
# groups_dfs, dfs_groups = group_alarms_by(df, ['srid', 'depth', 'corners', 'target'], ['lane'])

#

# print(cross_val_folds)

# dataset = AlarmDataset(
#     csv_file='data/three_region_cross_val.csv',
#     root_dir='data/',
#     transform=transforms.ToTensor()
# )
#
# for train_idx, validation_idx in KFold(n_splits=10).split(np.zeros((len(dataset), 1))):  # (samples, features)
#     train_sampler = SubsetRandomSampler(train_idx)
#     validation_sampler = SubsetRandomSampler(validation_idx)
#
#     train_loader = DataLoader(
#         dataset,
#         batch_size=1,
#         sampler=train_sampler
#     )
#
#     validation_loader = DataLoader(
#         dataset,
#         batch_size=2,
#         sampler=validation_sampler
#     )
#     for epoch in range(2):
#         for batch_index, (inputs, label) in enumerate(train_loader):
#             print(epoch, batch_index, label)
#
#     for epoch in range(2):
#         for batch_index, (inputs, labels) in enumerate(validation_loader):
#             print(epoch, batch_index, labels)
#
# def convert_utm_to_lat_lon(df):
#     # Define the two projections.
#     # p1 = pyproj.Proj(init='eps:32618')
#     p1 = pyproj.Proj('+proj=utm +zone=18 +datum=WGS84 +units=m +no_defs')
#     # p2 = pyproj.Proj(init='eps:32601')
#     p2 = pyproj.Proj('+proj=utm +zone=1 +datum=WGS84 +units=m +no_defs')
#     # p3 = pyproj.Proj(init='eps:32611')
#     p3 = pyproj.Proj('+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs')
#
#     df
