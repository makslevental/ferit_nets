import os
from collections import defaultdict, Counter
from operator import itemgetter
from typing import List, Tuple, Optional, Dict, Union
from sklearn.model_selection import (StratifiedKFold, KFold, GroupKFold, BaseCrossValidator, GroupShuffleSplit,
                                     StratifiedShuffleSplit, ShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold)
from rtree import index as r_index
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
    df['utm'] = df[['xUTM', 'yUTM']].apply(lambda u: (u.xUTM, u.yUTM), axis=1)
    df.drop(['xUTM', 'yUTM'], inplace=True, axis=1)

    df.drop(
        [
            'sensor_type', 'dt', 'ch', 'scan', 'time', 'MineIsInsideLane', 'DIST', 'objectdist',
            'category', 'isDetected', 'IsInsideLane', 'prescreener', 'known_obj', 'event_type', 'id', 'lsd_kv',
            'encoder', 'event_timestamp', 'fold'
        ],
        inplace=True,
        axis=1
    )

    return df


AlarmGroupId = Union[Tuple, int]
GroupedAlarms = Tuple[AlarmGroupId, List[pd.DataFrame]]
AlarmGroups = Tuple[List[GroupedAlarms], List[Tuple[int, AlarmGroupId]]]

CrossValSplit = Tuple[pd.DataFrame, pd.DataFrame]


def compute_cross_val_stats(cross_vals: List[CrossValSplit], attrs: List[str] = None):
    if attrs is None: attrs = ['srid', 'target', 'depth', 'corners']
    for train, test in cross_vals:
        train = train.sort_values(attrs, na_position='last')
        train_gb = train.groupby(attrs, sort=False)

        test = test.sort_values(attrs, na_position='last')
        test_gb = test.groupby(attrs, sort=False)

        train_gb_size = train_gb.size().to_frame('counts')
        test_gb_size = test_gb.size().to_frame('counts')

        train_gb_size['counts'] = train_gb_size['counts'].map(lambda x: x / sum(train_gb_size['counts']))
        test_gb_size['counts'] = test_gb_size['counts'].map(lambda x: x / sum(test_gb_size['counts']))

        for level in train_gb_size.index.levels[0]:
            for sublevel in train_gb_size.index.levels[1]:
                for subsublevel in train_gb_size.index.levels[2]:
                    print('train', level, sublevel, subsublevel, train_gb_size.T[level, sublevel].loc['counts'].values)
                    print('test', level, sublevel, subsublevel, test_gb_size.T[level, sublevel].loc['counts'].values)
                    print()


def group_alarms_by(alarms: pd.DataFrame,
                    attrs: List[str] = None,
                    group_attrs: List[str] = None
                    ) -> AlarmGroups:
    if attrs is None: attrs = ['srid', 'target', 'depth', 'corners']

    if group_attrs:
        assert set(group_attrs) <= set(attrs)
    group_idx_attr_map = {attr: i for i, attr in enumerate(attrs)}

    alarms.sort_values(attrs, inplace=True, na_position='last')

    gb = alarms.groupby(attrs, sort=False)
    groups = list(gb)

    if group_attrs:
        group_df_map = defaultdict(list)
        for group_tuple, group in groups:
            group_id = tuple([group_tuple[group_idx_attr_map[group_attr]] for group_attr in group_attrs])

            if DEBUG:
                group['corners'] = group['corners'].map(list, na_action='ignore')

            group_df_map[group_id].append(group)
        groups_dfs = list(group_df_map.items())

    else:
        groups_dfs = [(groupid, [alarms]) for groupid, (_, alarms) in enumerate(groups)]

    dfidxs_groups = []
    for groupid, dfs in groups_dfs:
        for df in dfs:
            dfidxs_groups.extend([(idx, groupid) for idx in df.index])

    if DEBUG:
        alarms['corners'] = alarms['corners'].map(list, na_action='ignore')

    return groups_dfs, dfidxs_groups


def create_cross_val_splits(n_splits: int,
                            cv: BaseCrossValidator,
                            all_alarms: pd.DataFrame,
                            dfs_groups: List[Tuple[int, AlarmGroupId]],
                            groups=None
                            ) -> List[CrossValSplit]:
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

        alarm_splits.append(
            (all_alarms.loc[train_idxs],
             all_alarms.loc[test_idxs])
        )

    return alarm_splits


def group_false_alarms(all_false_alarms: pd.DataFrame) -> AlarmGroups:
    false_alarm_groups, _ = group_alarms_by(all_false_alarms, attrs=['srid', 'site'], group_attrs=['site'])

    groups_dfs = []
    dfidxs_groups = []
    # current group id is probably site
    for groupid, dfs in false_alarm_groups:
        for df in dfs:
            r_idx = r_index.Index()
            for dfidx, utm in df['utm'].iteritems():
                r_idx.insert(dfidx, utm + utm)  # hack to insert a point

            for (leaf_id, dfidxs, coords) in r_idx.leaves():
                groups_dfs.append(
                    ((tuple(coords[:2]), f'{"_".join(groupid)}_miss_{leaf_id}'), all_false_alarms.loc[dfidxs])
                )

                dfidxs_groups.extend(
                    [(dfidx, (tuple(coords[:2]), f'{"_".join(groupid)}_miss_{leaf_id}')) for dfidx in dfidxs]
                )

    return groups_dfs, dfidxs_groups


def join_cross_val_splits(cv_splits_1: List[CrossValSplit],
                          alarms_1: pd.DataFrame,
                          groups_dfs_1: List[GroupedAlarms],
                          dfs_groups_1: List[Tuple[int, AlarmGroupId]],
                          cv_splits_2: List[CrossValSplit],
                          alarms_2: pd.DataFrame,
                          groups_dfs_2: List[GroupedAlarms],
                          dfs_groups_2: List[Tuple[int, AlarmGroupId]]
                          ) -> Tuple[
    List[CrossValSplit], pd.DataFrame, List[GroupedAlarms], List[Tuple[int, AlarmGroupId]]]:
    common_groupdids = set(map(itemgetter(0), groups_dfs_1)) & set(map(itemgetter(0), groups_dfs_2))
    assert not common_groupdids, f'groupids collision {common_groupdids}'

    common_dfidxs = set(map(itemgetter(0), dfs_groups_1)) & set(map(itemgetter(0), dfs_groups_2))
    assert not common_dfidxs, f'dfidx collision {common_dfidxs}'

    cv_splits = []
    for cv_11, cv_12 in zip(cv_splits_1, cv_splits_2):
        concat_split = map(pd.concat, zip(cv_11, cv_12))
        for cp in concat_split:
            cp.index = cp.index.map(int)
        cv_splits.append(
            concat_split
        )

    concat_alarms = pd.concat([alarms_1, alarms_2])
    concat_alarms.index = concat_alarms.index.map(int)

    return (
        cv_splits,
        concat_alarms,
        groups_dfs_1 + groups_dfs_2,
        dfs_groups_1 + dfs_groups_2
    )


def visualize_cross_vals(alarms: pd.DataFrame, n_splits=10):
    true_alarms = alarms[alarms['HIT'] == 1].copy(deep=True)
    false_alarms = alarms[alarms['HIT'] == 0].copy(deep=True)
    del alarms

    ### these functions mutate the df passed in (sort and reset index after sort)
    true_alarm_groups_dfs, true_alarm_dfs_groups = group_alarms_by(true_alarms, group_attrs=['target', 'depth'])
    false_alarm_groups_dfs, false_alarm_dfs_groups = group_false_alarms(false_alarms)

    visualize_groups(true_alarm_dfs_groups, true_alarms)
    visualize_groups(false_alarm_dfs_groups, false_alarms)

    cvs = [
        KFold(n_splits=n_splits),
        # GroupKFold(n_splits=n_splits),
        # ShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=0),
        StratifiedKFold(n_splits=n_splits, shuffle=False),
        # GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0),
        # StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0),
        # RepeatedKFold(n_splits=n_splits, n_repeats=2, random_state=0),
        # RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=2, random_state=0),
    ]

    for cv in cvs:
        true_alarm_cross_val_splits = create_cross_val_splits(n_splits, cv, true_alarms, true_alarm_dfs_groups)
        false_alarm_cross_val_splits = create_cross_val_splits(n_splits, cv, false_alarms, false_alarm_dfs_groups)

        # compute_cross_val_stats(true_alarm_cross_val_splits)
        # compute_cross_val_stats(false_alarm_cross_val_splits, attrs=['srid', 'site'])

        cross_val_splits, alarms, groups_dfs, dfs_groups = join_cross_val_splits(
            true_alarm_cross_val_splits, true_alarms, true_alarm_groups_dfs, true_alarm_dfs_groups,
            false_alarm_cross_val_splits, false_alarms, false_alarm_groups_dfs, false_alarm_dfs_groups,
        )

        plot_cv_indices(cv, cross_val_splits, dfs_groups, alarms)


root = os.getcwd()
# tuf_table_file_name = 'small_maxs_table.csv'
# tuf_table_file_name = 'medium_maxs_table.csv'
# tuf_table_file_name = 'big_maxs_table.csv'
tuf_table_file_name = 'all_maxs.csv'
all_alarms = tuf_table_csv_to_df(os.path.join(root, tuf_table_file_name))

visualize_cross_vals(all_alarms, n_splits=10)
