import os
from collections import defaultdict, Counter
from operator import itemgetter
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import (StratifiedKFold, KFold, GroupKFold, BaseCrossValidator, GroupShuffleSplit,
                                     StratifiedShuffleSplit, ShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold)
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from visualization import plot_cv_indices, visualize_groups

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
        corner_tuples = tuple(zip(corners[:num_corners // 2], corners[num_corners // 2:]))
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
        all_alarms: pd.DataFrame,
        attrs: List[str] = None,
        group_attrs: List[str] = None
) -> Tuple[List[Tuple[int, pd.DataFrame]], List[Tuple[int, int]]]:
    if attrs is None: attrs = ['srid', 'target', 'depth', 'corners']

    if group_attrs:
        assert set(group_attrs) < set(attrs)
    group_idx_attr_map = {attr: i for i, attr in enumerate(attrs)}

    all_alarms.sort_values(attrs, inplace=True, na_position='last')
    all_alarms.reset_index(inplace=True)

    gb = all_alarms.dropna(subset=['corners']).groupby(attrs, sort=False)
    print(gb.size())
    groups = list(gb)
    misses = all_alarms[all_alarms['HIT'] == 0]

    if group_attrs:
        group_df_map = defaultdict(list)
        for group_tuple, group in groups:
            group_id = tuple([group_tuple[group_idx_attr_map[group_attr]] for group_attr in group_attrs])

            if DEBUG:
                group['corners'] = group['corners'].map(list, na_action='ignore')

            group_df_map[group_id].append(group)
        groups_dfs = list(group_df_map.items())
        groups_dfs.append((('miss',) * len(group_id), np.split(misses, len(misses))))
    else:
        groups_dfs = [(group, [all_alarms]) for group, (_, all_alarms) in enumerate(groups)]
        groups_dfs.append((len(groups_dfs), np.split(misses, len(misses))))

    dfidx_groups = []
    for group, dfs in groups_dfs:
        for df in dfs:
            dfidx_groups.extend([(idx, group) for idx in df.index])

    if DEBUG:
        all_alarms['corners'] = all_alarms['corners'].map(list, na_action='ignore')

    return groups_dfs, dfidx_groups


def create_cross_val_splits(
        n_splits: int,
        cv: BaseCrossValidator,
        all_alarms: pd.DataFrame,
        dfs_groups: List[Tuple[int, int]],
        groups=None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    group_sizes = Counter(map(itemgetter(1), dfs_groups))
    # if number of instances/grouping is less than the number of folds then you can't distribute across the folds evenly

    enough = [(df_idx, group) for df_idx, group in dfs_groups if group_sizes[group] >= n_splits]
    not_enough = [(df_idx, group) for df_idx, group in dfs_groups if group_sizes[group] < n_splits]

    if getattr(cv, 'shuffle', None):
        # in place
        np.random.shuffle(not_enough)

    not_enough_splits = np.array_split(
        map(itemgetter(0), not_enough),
        n_splits
    )

    enough_xs = np.asarray(map(itemgetter(0), enough))
    enough_groups = map(lambda e: '_'.join(map(str, e[1])), enough)
    alarm_splits = []
    cv_iter = cv.split(
        X=enough_xs,
        y=enough_groups,
        groups=groups or enough_groups
    )
    # these are indices in enough_group_xs not in the dataframe (because that's how sklearn works)
    for i, (enough_train_idxs, enough_test_idxs) in enumerate(cv_iter):
        # these are now indices in the dataframe
        enough_train_df_idxs, enough_test_df_idxs = enough_xs[enough_train_idxs], enough_xs[enough_test_idxs]

        j = i % n_splits
        not_enough_group_train_idxs = np.hstack(not_enough_splits[:j] + not_enough_splits[j + 1:])
        not_enough_group_test_idxs = not_enough_splits[j]

        assert not set(enough_train_df_idxs) & set(enough_test_df_idxs)
        assert not set(not_enough_group_train_idxs) & set(not_enough_group_test_idxs)

        train_idxs = np.hstack([enough_train_df_idxs, not_enough_group_train_idxs])
        test_idxs = np.hstack([enough_test_df_idxs, not_enough_group_test_idxs])

        assert not set(train_idxs) & set(test_idxs)

        # alarm_splits.append(
        #     (all_alarms.loc[enough_train_df_idxs],
        #      all_alarms.loc[enough_test_df_idxs])
        # )
        alarm_splits.append(
            (all_alarms.loc[train_idxs],
             all_alarms.loc[test_idxs])
        )

    return alarm_splits


def visualize_cross_vals(all_alarms: pd.DataFrame, n_splits=10):
    _groups_dfs, dfs_groups = group_alarms_by(all_alarms, group_attrs=['target', 'depth'])
    visualize_groups(dfs_groups, all_alarms)

    cvs = [
        # KFold(n_splits=n_splits),
        # GroupKFold(n_splits=n_splits),
        # ShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=0),
        StratifiedKFold(n_splits=n_splits, shuffle=False),
        # GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0),
        # StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0),
        # RepeatedKFold(n_splits=n_splits, n_repeats=2, random_state=0),
        # RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=2, random_state=0),
    ]

    for cv in cvs:
        cross_val_splits = create_cross_val_splits(n_splits, cv, all_alarms, dfs_groups)
        plot_cv_indices(cv, cross_val_splits, dfs_groups, all_alarms)


root = os.getcwd()
tuf_table_file_name = 'three_maxs_table.csv'
tuf_table_file_name = 'medium_maxs_table.csv'
tuf_table_file_name = 'big_maxs_table.csv'
# tuf_table_file_name = 'all_maxs.csv'
all_alarms = tuf_table_csv_to_df(os.path.join(root, tuf_table_file_name))
# df['corners'] = df['corners'].map(tuple, na_action='ignore')
# attrs = ['srid', 'target', 'depth', 'corners']

# _groups_dfs, dfs_groups = group_alarms_by(df, attrs)

visualize_cross_vals(all_alarms, n_splits=10)
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


# region hold out target stratified

# normalize names

# depth and name as label
