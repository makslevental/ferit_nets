import itertools
import logging
import operator
import os
from collections import defaultdict, Counter
from functools import reduce
from operator import itemgetter
from pprint import PrettyPrinter
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from rtree import index as r_index
from shapely.geometry import MultiPoint
from sklearn.model_selection import (StratifiedKFold, BaseCrossValidator, LeaveOneGroupOut)

from bro_nets import DEBUG
from bro_nets.util import *
from bro_nets.visualization import plot_cv_indices, visualize_groups

pp = PrettyPrinter(indent=2)

logger = logging.Logger(__name__)


def tuf_table_csv_to_df(fp) -> pd.DataFrame:
    df = pd.read_csv(fp)

    df = df.where((pd.notnull(df)), None)

    def crnrs_to_tpl(crnrs: pd.DataFrame) -> Optional[List[Tuple]]:
        crnrs = filter(None, crnrs)
        n_crnrs = len(crnrs)
        crnr_tpls = tuple(zip(crnrs[:n_crnrs // 2], crnrs[n_crnrs // 2:]))
        return crnr_tpls if len(crnr_tpls) else None

    crnrs_cols = filter(lambda x: 'corner_' in x, df.columns)

    if len(crnrs_cols):
        df['corners'] = df[crnrs_cols].apply(crnrs_to_tpl, axis=1)
        df.drop(crnrs_cols, inplace=True, axis=1)

    if 'xUTM' in df.columns and 'yUTM' in df.columns:
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
                    groupby_attrs: List[str] = None,
                    group_assign_attrs: List[str] = None
                    ) -> Tuple[AlarmGroups, pd.DataFrame]:
    if groupby_attrs is None: groupby_attrs = ['srid', 'target', 'depth', 'corners']

    alarms = alarms.copy(deep=True)

    assert set(groupby_attrs) <= set(alarms.columns)

    if 'corners' in groupby_attrs and isinstance(alarms.loc[0]['corners'], list):
        raise Exception("didn't cast corners to tuples")

    if group_assign_attrs:
        assert set(group_assign_attrs) <= set(groupby_attrs), \
            f'group assign attrs not in groupby attrs {group_assign_attrs} {groupby_attrs}'

    alarms.sort_values(groupby_attrs, inplace=True, na_position='last')
    alarms.reset_index(inplace=True, drop=True)

    assert not alarms[groupby_attrs].isnull().values.any(), f'null values in table in groupby columns {groupby_attrs}'

    gb = alarms.groupby(groupby_attrs, sort=False)
    groups = list(gb)

    if group_assign_attrs:
        groupids_dfidxs = []
        groupidx_attr_map = {attr: i for i, attr in enumerate(groupby_attrs)}
        for groupid_tuple, group in groups:
            groupid = tuple(
                groupid_tuple[groupidx_attr_map[group_attr]] for group_attr in group_assign_attrs
            )
            groupids_dfidxs.append((groupid, group.index))
    else:
        groupids_dfidxs = [(groupid, group.index) for groupid, (_, group) in enumerate(groups)]

    dfidxs_groupids = []
    for groupid, dfidxs in groupids_dfidxs:
        dfidxs_groupids.extend([(idx, groupid) for idx in dfidxs])

    assert len(alarms) == sum(map(len, map(itemgetter(1), groupids_dfidxs))), \
        f'mismatched size {len(alarms), sum(map(len, map(itemgetter(1), groupids_dfidxs)))}'
    assert len(alarms) == len(dfidxs_groupids), f'mismatched size {len(alarms), len(dfidxs_groupids)}'
    assert all(map(itemgetter(0), dfidxs_groupids) == alarms.index)

    return (groupids_dfidxs, dfidxs_groupids), alarms


def create_cross_val_splits(n_splits: int,
                            cv: BaseCrossValidator,
                            alarms: pd.DataFrame,
                            dfidxs_grpids: List[Tuple[int, AlarmGroupId]],
                            groups=None
                            ) -> List[CrossValSplit]:
    grp_szs = Counter(map(itemgetter(1), dfidxs_grpids))

    train = [(df_idx, groupid) for df_idx, groupid in dfidxs_grpids if grp_szs[groupid] >= n_splits]

    if len(train) == 0:
        raise Exception(f'no groups with at least {n_splits} alarms: {grp_szs}')
    else:
        train_xs = np.asarray(map(itemgetter(0), train))
        train_groups = np.asarray(map(lambda e: '_'.join(map(str, e[1])), train))
        cv_iter = cv.split(
            X=train_xs,
            y=train_groups,
            groups=train_groups
        )

    # if number of instances/grouping is less than the number of folds then you can't distribute across the folds evenly
    not_enough = [(df_idx, groupid) for df_idx, groupid in dfidxs_grpids if grp_szs[groupid] < n_splits]
    if len(not_enough) > 0:
        not_enough_xs = np.asarray(map(itemgetter(0), not_enough))
        if getattr(cv, 'shuffle', False): np.random.shuffle(not_enough_xs)
        len_t = len(train_xs)
        assert not set(train_xs) & set(not_enough_xs)
        train_xs = np.hstack([train_xs, not_enough_xs])

        splits = np.array_split(range(len(not_enough_xs)), n_splits)
        if getattr(cv, 'shuffle', False): np.random.shuffle(splits)

        cv_iter = (
            # indices from not_enough actually point to behind/infront/whatever you want to call it
            # of train_xs, so increment them all by len_t (the length of train_xs)
            (np.hstack([tr1, len_t + tr2]), np.hstack([tt1, len_t + tt2])) for ((tr1, tt1), (tr2, tt2)) in
            zip(cv_iter, ((np.hstack(splits[:i] + splits[i + 1:]), splits[i]) for i in range(n_splits)))
        )

    alarm_splits = []
    # these are indices in enough_group_xs not in the dataframe (because that's how sklearn works)
    for i, (train_idxs, test_idxs) in enumerate(cv_iter):
        # these are now indices in the dataframe
        train_dfidxs, test_dfidxs = train_xs[train_idxs], train_xs[test_idxs]

        assert not set(train_dfidxs) & set(test_dfidxs)

        train_df = alarms.loc[train_dfidxs]
        train_df.index = train_df.index.map(int)
        test_df = alarms.loc[test_dfidxs]
        test_df.index = test_df.index.map(int)

        alarm_splits.append((train_df, test_df))

    return alarm_splits


def group_false_alarms(all_false_alarms: pd.DataFrame) -> Tuple[AlarmGroups, pd.DataFrame]:
    (falrmgrpids_dfidxs, _), grpdfalrmsdf = group_alarms_by(
        all_false_alarms,
        groupby_attrs=['srid', 'site'],
        group_assign_attrs=['site']
    )

    grpids_dfidxs = []
    dfidxs_grpids = []
    for groupid, dfidxs in falrmgrpids_dfidxs:
        instance_rows_utms = grpdfalrmsdf.loc[dfidxs]['utm']
        r_idx = r_index.Index()
        groupid = (f'{"_".join(groupid)}_miss',)

        for instance_dfidx, utm in instance_rows_utms.iteritems():
            r_idx.insert(
                instance_dfidx,
                utm + utm  # hack to insert a point
            )

        grouped_idxs = set()
        near_dfidxs = []

        for dfidx, utm in instance_rows_utms.iteritems():
            if dfidx not in grouped_idxs:
                # 0,0, 60, 60, minx, miny, maxx, maxy
                utmx, utmy = utm

                near_points_idxs = set(r_idx.intersection((utmx - 5, utmy - 5, utmx + 5, utmy + 5)))

                ungrouped_idxs = near_points_idxs - grouped_idxs

                grouped_idxs.update(near_points_idxs)
                near_dfidxs.append(list(ungrouped_idxs))

                dfidxs_grpids.extend(
                    [(dfidx, groupid) for dfidx in ungrouped_idxs]
                )

                near_utms = instance_rows_utms.loc[ungrouped_idxs].values
                if DEBUG:
                    logger.debug(
                        f'convex hull area for {"_".join(groupid)}_miss: {MultiPoint(near_utms).convex_hull.area}'
                    )

        grpids_dfidxs.append((groupid, pd.Index(near_dfidxs)))

    assert len(all_false_alarms) == sum(map(lambda g: sum(map(len, g[1])), grpids_dfidxs)), \
        f'mismatched size {len(all_false_alarms), sum(map(lambda g: sum(map(len, g[1])), grpids_dfidxs))}'
    assert len(all_false_alarms) == len(dfidxs_grpids), \
        f'mismatched size {len(all_false_alarms), len(dfidxs_grpids)}'

    for groupid, dfidxs in grpids_dfidxs:
        logger.info(f'group {groupid} # alarms {len(dfidxs)}')

    return (grpids_dfidxs, dfidxs_grpids), grpdfalrmsdf


def join_cross_val_splits(cv_splits_1: List[CrossValSplit],
                          alarms_1: pd.DataFrame,
                          dfidxs_groups_1: List[Tuple[int, AlarmGroupId]],
                          cv_splits_2: List[CrossValSplit],
                          alarms_2: pd.DataFrame,
                          dfidxs_groups_2: List[Tuple[int, AlarmGroupId]]
                          ) -> Tuple[
    List[CrossValSplit], pd.DataFrame, List[GroupedAlarms], List[Tuple[int, AlarmGroupId]]]:
    assert len(cv_splits_1) == len(cv_splits_2)

    concat_alrms = pd.concat(
        [alarms_1, alarms_2],
        keys=['alarms_1', 'alarms_2'],
        names=['alarms', 'old_index']
    )

    alrms_1_new_idx = concat_alrms.index.get_locs([['alarms_1'], alarms_1.index])
    alrms_2_new_idx = concat_alrms.index.get_locs([['alarms_2'], alarms_2.index])

    dfidxs_grpids_1_map = dict(dfidxs_groups_1)
    dfidxs_grpids_2_map = dict(dfidxs_groups_2)

    new_dfidxs_grpids = []
    new_grpids_dfidxs = defaultdict(lambda: pd.Index([]))

    for alrm_new_idx, alrm_old_idx, dfidx_groupids_map in [(alrms_1_new_idx, alarms_1.index, dfidxs_grpids_1_map),
                                                           (alrms_2_new_idx, alarms_2.index, dfidxs_grpids_2_map)]:
        for new_idx, old_idx in zip(alrm_new_idx, alrm_old_idx):
            grp_id = dfidx_groupids_map[old_idx]
            new_dfidxs_grpids.append((int(new_idx), grp_id))
            new_grpids_dfidxs[grp_id] = new_grpids_dfidxs[grp_id].append(pd.Index([new_idx]))

    cv_splits = []
    for i in range(len(cv_splits_1)):
        train_1, test_1 = cv_splits_1[i]
        train_2, test_2 = cv_splits_2[i]

        new_train_1_idx = concat_alrms.index.get_locs([['alarms_1'], train_1.index])
        new_test_1_idx = concat_alrms.index.get_locs([['alarms_1'], test_1.index])

        new_train_2_idx = concat_alrms.index.get_locs([['alarms_2'], train_2.index])
        new_test_2_idx = concat_alrms.index.get_locs([['alarms_2'], test_2.index])

        train, test = pd.concat([train_1, train_2]), pd.concat([test_1, test_2], ignore_index=True)
        train.index = pd.Index(np.concatenate([new_train_1_idx, new_train_2_idx]))
        test.index = pd.Index(np.concatenate([new_test_1_idx, new_test_2_idx]))

        cv_splits.append((train, test))

    return (
        cv_splits,
        concat_alrms.reset_index(drop=True),
        reduce(
            operator.concat,
            map(
                lambda gid_dfidxs: [*itertools.product(['_'.join(map(str, gid_dfidxs[0]))], gid_dfidxs[1])],
                new_grpids_dfidxs.items()
            ),
        ),
        new_dfidxs_grpids
    )


def split_t_f_alarms(all_alarms: pd.DataFrame):
    true_alarms = all_alarms[all_alarms['HIT'] == 1].copy(deep=True)
    true_alarms.reset_index(inplace=True, drop=True)
    false_alarms = all_alarms[all_alarms['HIT'] == 0].copy(deep=True)
    false_alarms.reset_index(inplace=True, drop=True)
    return true_alarms, false_alarms


def _splits(cv, alarms: pd.DataFrame, n_splits, attrs, group_attrs, separate_t_f=True) -> Tuple[
    List[CrossValSplit], pd.DataFrame, List[GroupedAlarms], List[Tuple[int, AlarmGroupId]]
]:
    if separate_t_f:
        t_alrms_df, f_alrms_df = split_t_f_alarms(alarms)

        (t_alrm_grpids_dfidxs, t_alrm_dfidxs_grpids), grpd_t_alrms_df = group_alarms_by(t_alrms_df,
                                                                                        groupby_attrs=attrs,
                                                                                        group_assign_attrs=group_attrs
                                                                                        )
        (f_alrm_grpids_dfidxs, f_alrm_dfidxs_grpids), grpd_f_alrms_df = group_false_alarms(f_alrms_df)

        t_crs_val_splits = create_cross_val_splits(n_splits, cv, grpd_t_alrms_df, t_alrm_dfidxs_grpids)
        f_crs_val_splits = create_cross_val_splits(n_splits, cv, grpd_f_alrms_df, f_alrm_dfidxs_grpids)

        crs_val_splits, alarms, grids_dfidxs, dfidxs_grpids = join_cross_val_splits(
            t_crs_val_splits, grpd_t_alrms_df, t_alrm_dfidxs_grpids,
            f_crs_val_splits, grpd_f_alrms_df, f_alrm_dfidxs_grpids,
        )
    else:
        (grids_dfidxs, dfidxs_grpids), grpd_alrms_df = group_alarms_by(alarms,
                                                                       groupby_attrs=attrs,
                                                                       group_assign_attrs=group_attrs
                                                                       )
        crs_val_splits = create_cross_val_splits(n_splits, cv, grpd_alrms_df, dfidxs_grpids)

    return crs_val_splits, alarms, grids_dfidxs, dfidxs_grpids


def logo_region_splits(alarms: pd.DataFrame) -> Tuple[
    List[CrossValSplit], pd.DataFrame, List[GroupedAlarms], List[Tuple[int, AlarmGroupId]]
]:
    cv = LeaveOneGroupOut()
    return _splits(cv, alarms, n_splits=10, attrs=['srid', 'site', 'lane'], group_attrs=['lane'],
                   separate_t_f=False)


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
    rgn_splits, rgn_alarms, rgn_groups_dfs, rgn_dfs_groups = logo_region_splits(alarms)
    if DEBUG:
        vis_crossv_folds(rgn_splits, rgn_dfs_groups, rgn_alarms, 'region logo fold')

    for i, (rgn_train, rgn_test) in enumerate(rgn_splits):
        strat_splits, strat_alarms, strat_groups_dfs, strat_dfs_groups = kfold_stratified_target_splits(
            rgn_train, n_splits_stratified
        )
        if DEBUG:
            vis_crossv_folds(strat_splits, strat_dfs_groups, strat_alarms,
                             f'strat {n_splits_stratified} folds region fold {i}')
        yield strat_splits, rgn_train, rgn_test


if __name__ == '__main__':
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    root = os.getcwd()
    # tuf_table_file_name = 'small_maxs_table.csv'
    # tuf_table_file_name = 'medium_maxs_table.csv'
    tuf_table_file_name = 'big_maxs_table.csv'
    # tuf_table_file_name = 'bigger_maxs_table.csv'
    # tuf_table_file_name = 'even_bigger_maxs_table.csv'
    # tuf_table_file_name = 'all_maxs.csv'
    all_alarms = tuf_table_csv_to_df(os.path.join(root, tuf_table_file_name))

    for _ in region_and_stratified(all_alarms, n_splits_region=3, n_splits_stratified=30):
        continue

