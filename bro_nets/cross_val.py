import itertools
import operator

import os
from collections import defaultdict, Counter
from functools import reduce
from operator import itemgetter
from typing import List, Tuple, Optional, Dict, Union, NewType
from sklearn.model_selection import (StratifiedKFold, KFold, GroupKFold, BaseCrossValidator, GroupShuffleSplit,
                                     StratifiedShuffleSplit, ShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold,
                                     LeaveOneGroupOut)
from rtree import index as r_index
import numpy as np
import pandas as pd
from bro_nets.visualization import plot_cv_indices, visualize_groups
from pprint import PrettyPrinter
from shapely.geometry import MultiPoint

from bro_nets.util import *
from bro_nets import DEBUG

import logging

pp = PrettyPrinter(indent=2)


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
        set(df.columns) &
        {
            'sensor_type', 'dt', 'ch', 'scan', 'time', 'MineIsInsideLane', 'DIST', 'objectdist',
            'category', 'isDetected', 'IsInsideLane', 'prescreener', 'known_obj', 'event_type', 'id', 'lsd_kv',
            'encoder', 'event_timestamp', 'fold'
        },
        inplace=True,
        axis=1
    )

    return df


AlarmGroupId = Union[Tuple, int]
GroupedAlarms = Tuple[AlarmGroupId, pd.Int64Index]
AlarmGroups = Tuple[List[GroupedAlarms], List[Tuple[int, AlarmGroupId]]]

CrossValSplit = Tuple[pd.DataFrame, pd.DataFrame]


def group_alarms_by(alarms: pd.DataFrame,
                    attrs: List[str] = None,
                    group_attrs: List[str] = None
                    ) -> Tuple[AlarmGroups, pd.DataFrame]:
    if attrs is None: attrs = ['srid', 'target', 'depth', 'corners']

    alarms = alarms.copy(deep=True)

    assert set(attrs) <= set(alarms.columns)

    if 'corners' in attrs and isinstance(alarms.loc[0]['corners'], list):
        raise Exception("didn't cast corners to tuples")

    if group_attrs:
        assert set(group_attrs) <= set(attrs), f'group attr not in groupby attrs {group_attrs} {attrs}'
    group_idx_attr_map = {attr: i for i, attr in enumerate(attrs)}

    alarms.sort_values(attrs, inplace=True, na_position='last')
    alarms.reset_index(inplace=True, drop=True)

    assert not alarms[attrs].isnull().values.any(), f'null values in table in groupby columns {attrs}'

    gb = alarms.groupby(attrs, sort=False)
    groups = list(gb)

    if group_attrs:
        groups_dfidxs = []
        for group_tuple, group in groups:
            groupid = tuple([group_tuple[group_idx_attr_map[group_attr]] for group_attr in group_attrs])
            groups_dfidxs.append((groupid, group.index))
    else:
        groups_dfidxs = [(groupid, group.index) for groupid, (_, group) in enumerate(groups)]

    dfidxs_groups = []
    for groupid, dfidxs in groups_dfidxs:
        dfidxs_groups.extend([(idx, groupid) for idx in dfidxs])

    if DEBUG:
        alarms['corners'] = alarms['corners'].map(list, na_action='ignore')

    assert len(alarms) == sum(map(len, map(itemgetter(1), groups_dfidxs))), \
        f'mismatched size {len(alarms), sum(map(len, map(itemgetter(1), groups_dfidxs)))}'
    assert len(alarms) == len(dfidxs_groups), f'mismatched size {len(alarms), len(dfidxs_groups)}'
    assert all(map(itemgetter(0), dfidxs_groups) == alarms.index)

    return (groups_dfidxs, dfidxs_groups), alarms


def create_cross_val_splits(n_splits: int,
                            cv: BaseCrossValidator,
                            alarms: pd.DataFrame,
                            dfidxs_groups: List[Tuple[int, AlarmGroupId]],
                            groups=None
                            ) -> List[CrossValSplit]:
    group_sizes = Counter(map(itemgetter(1), dfidxs_groups))

    # if number of instances/grouping is less than the number of folds then you can't distribute across the folds evenly
    not_enough = [(df_idx, groupid) for df_idx, groupid in dfidxs_groups if group_sizes[groupid] < n_splits]
    if getattr(cv, 'shuffle', None): np.random.shuffle(not_enough)
    not_enough_splits = np.array_split(map(itemgetter(0), not_enough), n_splits)

    if len(not_enough):
        not_enough_counts = ",".join([str(gid) for gid in group_sizes if group_sizes[gid] < n_splits])
        logging.info(f'groups with not enough for n={n_splits}: {not_enough_counts}')

    enough = [(df_idx, groupid) for df_idx, groupid in dfidxs_groups if group_sizes[groupid] >= n_splits]
    enough_xs = np.asarray(map(itemgetter(0), enough))
    enough_groups = np.asarray(map(lambda e: '_'.join(map(str, e[1])), enough))
    if len(set(enough_groups)) == 0:
        raise Exception(f'no groups with {n_splits}: {group_sizes}')
    elif len(set(enough_groups)) < n_splits:
        warn_msg = f'not enough groups {len(set(enough_groups))} for splits {n_splits}\n using kfold \n'
        logging.warning(warn_msg)
        cv_iter = KFold(n_splits=n_splits).split(enough_xs)
    else:
        cv_iter = cv.split(
            X=enough_xs,
            y=enough_groups,
            groups=groups or enough_groups
        )

    alarm_splits = []
    # these are indices in enough_group_xs not in the dataframe (because that's how sklearn works)
    for i, (enough_train_idxs, enough_test_idxs) in enumerate(cv_iter):
        # these are now indices in the dataframe

        if DEBUG:
            train_counts, test_counts = Counter(enough_groups[enough_train_idxs]), Counter(
                enough_groups[enough_test_idxs])
            total_train, total_test = sum(train_counts.values()), sum(test_counts.values())
            for key in train_counts:
                train_counts[key] /= total_train

            for key in test_counts:
                test_counts[key] /= total_test

            for key in train_counts:
                logging.debug(key, train_counts[key] - test_counts[key])

        enough_train_df_idxs, enough_test_df_idxs = enough_xs[enough_train_idxs], enough_xs[enough_test_idxs]

        j = i % n_splits
        not_enough_group_train_idxs = np.hstack(not_enough_splits[:j] + not_enough_splits[j + 1:])
        not_enough_group_test_idxs = not_enough_splits[j]

        assert not set(enough_train_df_idxs) & set(not_enough_group_train_idxs)
        train_idxs = np.hstack([enough_train_df_idxs, not_enough_group_train_idxs])
        assert not set(enough_train_df_idxs) & set(not_enough_group_train_idxs)
        test_idxs = np.hstack([enough_test_df_idxs, not_enough_group_test_idxs])

        assert not set(train_idxs) & set(test_idxs)

        train_df = alarms.loc[train_idxs]
        train_df.index = train_df.index.map(int)
        test_df = alarms.loc[test_idxs]
        test_df.index = test_df.index.map(int)

        alarm_splits.append((train_df, test_df))

    return alarm_splits


def group_false_alarms(all_false_alarms: pd.DataFrame) -> Tuple[AlarmGroups, pd.DataFrame]:
    (false_alarm_groups, _), grouped_false_alarms = group_alarms_by(all_false_alarms, attrs=['srid', 'site'],
                                                                    group_attrs=['site'])

    # assert len(all_false_alarms) == sum(map(lambda g: sum(map(len, g[1])), false_alarm_groups)), \
    #     f'mismatched size {len(all_false_alarms), sum(map(lambda g: sum(map(len, g[1])), false_alarm_groups))}'

    groups_dfidxs = []
    dfidxs_groups = []
    # current group id is probably site
    for groupid, group_dfidxs in false_alarm_groups:
        instance_rows_utm = grouped_false_alarms.loc[group_dfidxs]['utm']
        r_idx = r_index.Index()
        groupid = (f'{"_".join(groupid)}_miss',)

        for instance_dfidx, utm in instance_rows_utm.iteritems():
            r_idx.insert(instance_dfidx, utm + utm)  # hack to insert a point

        grouped_idxs = set()
        near_dfidxs = []

        for dfidx, utm in instance_rows_utm.iteritems():
            if dfidx not in grouped_idxs:
                # 0,0, 60, 60, minx, miny, maxx, maxy
                utmx, utmy = utm

                near_points_idxs = set(r_idx.intersection((utmx - 5, utmy - 5, utmx + 5, utmy + 5)))

                ungrouped_idxs = near_points_idxs - grouped_idxs

                grouped_idxs.update(near_points_idxs)
                near_dfidxs.append(list(ungrouped_idxs))

                dfidxs_groups.extend(
                    [(dfidx, groupid) for dfidx in ungrouped_idxs]
                )

                near_utms = instance_rows_utm.loc[ungrouped_idxs].values
                logging.debug('convex hull area', f'{"_".join(groupid)}_miss',
                              MultiPoint(near_utms).convex_hull.area)

        groups_dfidxs.append((groupid, pd.Index(near_dfidxs)))

        # for (leaf_id, dfidxs, coords) in r_idx.leaves():
        #     dfidxs_groups.extend(
        #         [(dfidx, groupid) for dfidx in dfidxs]
        #     )
        #     near_dfidxs.append(dfidxs)
        #
        #     near_utms = df.loc[dfidxs]['utm'].values
        #     logging.debug('convex hull area', f'{"_".join(groupid)}_miss_{leaf_id}', MultiPoint(near_utms).convex_hull.area)

        # assert sum(map(len, near_dfidxs)) == len(dfidx), f'mismatch {sum(map(len, near_dfidxs)), len(dfidx)}'
        # assert len(grouped_idxs) == len(dfidx), f'mistmatch {sum(map(len, near_dfidxs)), len(dfidx)}'

    # assert len(all_false_alarms) == sum(map(lambda g: sum(map(len, g[1])), groups_dfidxs)), \
    #     f'mismatched size {len(all_false_alarms), sum(map(lambda g: sum(map(len, g[1])), groups_dfidxs))}'
    # assert len(all_false_alarms) == len(dfidxs_groups), f'mismatched size {len(all_false_alarms), len(dfidxs_groups)}'

    for groupid, group_dfidxs in groups_dfidxs:
        logging.debug(f'group {groupid} # alarms {len(group_dfidxs)}')

    return (groups_dfidxs, dfidxs_groups), grouped_false_alarms


# @jit(nopython=True)
def join_cross_val_splits(cv_splits_1: List[CrossValSplit],
                          alarms_1: pd.DataFrame,
                          dfidxs_groups_1: List[Tuple[int, AlarmGroupId]],
                          cv_splits_2: List[CrossValSplit],
                          alarms_2: pd.DataFrame,
                          dfidxs_groups_2: List[Tuple[int, AlarmGroupId]]
                          ) -> Tuple[
    List[CrossValSplit], pd.DataFrame, List[GroupedAlarms], List[Tuple[int, AlarmGroupId]]]:
    assert len(cv_splits_1) == len(cv_splits_2)

    # TODO: figure out whether there's a way to make this work (probably not because you lose index/group info
    # if not DEBUG:
    #     concat_alarms = pd.concat([alarms_1, alarms_2], ignore_index=True)
    #     cv_splits = []
    #     for i in range(len(cv_splits_1)):
    #         train_1, test_1 = cv_splits_1[i]
    #         train_2, test_2 = cv_splits_2[i]
    #         train, test = pd.concat([train_1, train_2], ignore_index=True), pd.concat([test_1, test_2],
    #                                                                                   ignore_index=True)
    #         cv_splits.append((train, test))
    #
    #     new_groups_dfidxs = {(i,): [None] for i in range(len(concat_alarms))}
    #     new_dfidxs_groups = [((i,), [None]) for i in range(len(concat_alarms))]
    #
    # return (
    #     cv_splits,
    #     concat_alarms.reset_index(drop=True),
    #     reduce(
    #         operator.concat,
    #         map(lambda g_d: [*itertools.product(['_'.join(map(str, g_d[0]))], g_d[1])], new_groups_dfidxs.items()),
    #     ),
    #     new_dfidxs_groups
    # )

    concat_alarms = pd.concat([alarms_1, alarms_2], keys=['alarms_1', 'alarms_2'],
                              names=['alarms', 'old_index'])

    alarms_1_new_idx = concat_alarms.index.get_locs([['alarms_1'], alarms_1.index])
    alarms_2_new_idx = concat_alarms.index.get_locs([['alarms_2'], alarms_2.index])

    dfidxs_groups_1_map = dict(dfidxs_groups_1)
    dfidxs_groups_2_map = dict(dfidxs_groups_2)

    new_dfidxs_groups = []
    new_groups_dfidxs = defaultdict(lambda: pd.Index([]))

    for new_idx, old_idx in zip(alarms_1_new_idx, alarms_1.index):
        new_dfidxs_groups.append((int(new_idx), dfidxs_groups_1_map[old_idx]))
        new_groups_dfidxs[dfidxs_groups_1_map[old_idx]] = new_groups_dfidxs[dfidxs_groups_1_map[old_idx]].append(
            pd.Index([new_idx]))

    for new_idx, old_idx in zip(alarms_2_new_idx, alarms_2.index):
        new_dfidxs_groups.append((int(new_idx), dfidxs_groups_2_map[old_idx]))
        new_groups_dfidxs[dfidxs_groups_2_map[old_idx]] = new_groups_dfidxs[dfidxs_groups_2_map[old_idx]].append(
            pd.Index([new_idx]))

    cv_splits = []
    for i in range(len(cv_splits_1)):
        train_1, test_1 = cv_splits_1[i]
        train_2, test_2 = cv_splits_2[i]

        new_train_1_idx = concat_alarms.index.get_locs([['alarms_1'], train_1.index])
        new_test_1_idx = concat_alarms.index.get_locs([['alarms_1'], test_1.index])

        new_train_2_idx = concat_alarms.index.get_locs([['alarms_2'], train_2.index])
        new_test_2_idx = concat_alarms.index.get_locs([['alarms_2'], test_2.index])

        train, test = pd.concat([train_1, train_2]), pd.concat([test_1, test_2], ignore_index=True)
        train.index = pd.Index(np.concatenate([new_train_1_idx, new_train_2_idx]))
        test.index = pd.Index(np.concatenate([new_test_1_idx, new_test_2_idx]))

        cv_splits.append((train, test))

    return (
        cv_splits,
        concat_alarms.reset_index(drop=True),
        reduce(
            operator.concat,
            map(lambda g_d: [*itertools.product(['_'.join(map(str, g_d[0]))], g_d[1])], new_groups_dfidxs.items()),
        ),
        new_dfidxs_groups
    )


def split_true_false_alarms(all_alarms: pd.DataFrame):
    true_alarms = all_alarms[all_alarms['HIT'] == 1].copy(deep=True)
    true_alarms.reset_index(inplace=True, drop=True)
    false_alarms = all_alarms[all_alarms['HIT'] == 0].copy(deep=True)
    false_alarms.reset_index(inplace=True, drop=True)
    return true_alarms, false_alarms


def _splits(cv, alarms: pd.DataFrame, n_splits, attrs, group_attrs) -> Tuple[
    List[CrossValSplit], pd.DataFrame, List[GroupedAlarms], List[Tuple[int, AlarmGroupId]]
]:
    true_alarms, false_alarms = split_true_false_alarms(alarms)

    (true_alarm_groups_dfs, true_alarm_dfs_groups), grouped_true_alarms = group_alarms_by(true_alarms,
                                                                                          attrs=attrs,
                                                                                          group_attrs=group_attrs)
    (false_alarm_groups_dfs, false_alarm_dfs_groups), grouped_false_alarms = group_false_alarms(false_alarms)

    true_alarm_cross_val_splits = create_cross_val_splits(n_splits, cv, grouped_true_alarms, true_alarm_dfs_groups)
    false_alarm_cross_val_splits = create_cross_val_splits(n_splits, cv, grouped_false_alarms, false_alarm_dfs_groups)

    cross_val_splits, alarms, groups_dfs, dfs_groups = join_cross_val_splits(
        true_alarm_cross_val_splits, grouped_true_alarms, true_alarm_dfs_groups,
        false_alarm_cross_val_splits, grouped_false_alarms, false_alarm_dfs_groups,
    )

    return cross_val_splits, alarms, groups_dfs, dfs_groups


def logo_region_splits(alarms: pd.DataFrame, n_splits=10) -> Tuple[
    List[CrossValSplit], pd.DataFrame, List[GroupedAlarms], List[Tuple[int, AlarmGroupId]]
]:
    cv = LeaveOneGroupOut()
    return _splits(cv, alarms, n_splits, attrs=['srid', 'lane', 'depth', 'corners'], group_attrs=['lane'])


def kfold_stratified_target_splits(alarms: pd.DataFrame, n_splits=10) -> Tuple[
    List[CrossValSplit], pd.DataFrame, List[GroupedAlarms], List[Tuple[int, AlarmGroupId]]
]:
    cv = StratifiedKFold(n_splits=n_splits)
    return _splits(cv, alarms, n_splits, attrs=['srid', 'target', 'depth', 'corners'], group_attrs=['target'])


def vis_crossv_folds(cross_val_splits: List[CrossValSplit], dfs_groups: List[Tuple[int, AlarmGroupId]],
                     alarms: pd.DataFrame, cv: str):
    visualize_groups(dfs_groups, alarms, 'classes/groups')
    plot_cv_indices(cross_val_splits, dfs_groups, alarms, cv)


def region_and_stratified(alarms: pd.DataFrame, n_splits_region, n_splits_stratified):
    rgn_splits, rgn_alarms, rgn_groups_dfs, rgn_dfs_groups = logo_region_splits(alarms, n_splits_region)
    if DEBUG:
        vis_crossv_folds(rgn_splits, rgn_dfs_groups, rgn_alarms, 'region logo fold')

    for i, (rgn_train, rgn_test) in enumerate(rgn_splits):
        strat_splits, strat_alarms, strat_groups_dfs, strat_dfs_groups = kfold_stratified_target_splits(
            rgn_train, n_splits_stratified
        )
        if DEBUG:
            vis_crossv_folds(strat_splits, strat_dfs_groups, strat_alarms,
                             f'strat {n_splits_stratified} folds region fold {i}')
        yield strat_splits, rgn_test


if __name__ == '__main__':
    root = os.getcwd()
    # tuf_table_file_name = 'small_maxs_table.csv'
    # tuf_table_file_name = 'medium_maxs_table.csv'
    # tuf_table_file_name = 'big_maxs_table.csv'
    tuf_table_file_name = 'bigger_maxs_table.csv'
    # tuf_table_file_name = 'even_bigger_maxs_table.csv'
    # tuf_table_file_name = 'all_maxs.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(root, tuf_table_file_name))
    region_and_stratified(all_alarms, n_splits_region=3, n_splits_stratified=10)

    # profile(kfold_region_splits, [all_alarms], {'n_splits': 3})
