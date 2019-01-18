import os.path as osp
import unittest
from operator import attrgetter as attrg

import numpy as np

from tuf_torch import cross_val
from tuf_torch.config import PROJECT_ROOT


class TestCrossVal(unittest.TestCase):
    csv_fp = osp.join(PROJECT_ROOT, "small_maxs_table.csv")

    def test_parse_csv(self):
        alarms = cross_val.tuf_table_csv_to_df(self.csv_fp)
        self.assertGreater(len(alarms), 0)
        self.assertTrue(
            (alarms.columns == ['HIT', 'event_id', 'sample', 'pre_conf', 'conf', 'lane', 'site', 'srid',
                                'target', 'depth', 'corners', 'utm']).all()
        )

        self.assertFalse(
            alarms[
                ['HIT', 'event_id', 'sample', 'pre_conf', 'conf', 'lane', 'site', 'srid', 'utm']
            ].isnull().any().any()
        )

        self.assertTrue(all(alarms['corners'].map(lambda c: isinstance(c, tuple) or c is None)))
        self.assertTrue((alarms['corners'].isnull() == alarms['HIT'].map(lambda h: h == 0)).all())
        self.assertTrue((alarms['depth'].isnull() == alarms['HIT'].map(lambda h: h == 0)).all())

    def test_group_alarms_by(self):
        alarms = cross_val.tuf_table_csv_to_df(self.csv_fp)
        with self.assertRaisesRegex(AssertionError, 'groupby'):
            cross_val.group_alarms_by(alarms, ['missing_columns'])
        with self.assertRaisesRegex(AssertionError, 'group assign'):
            cross_val.group_alarms_by(alarms, ['srid', 'target'], ['missing_group_attr'])

        true_alarms, false_alarms = cross_val.split_t_f_alarms(alarms)
        with self.assertRaisesRegex(AssertionError, 'null values'):
            cross_val.group_alarms_by(false_alarms)

        groupby_attrs = ['target', 'depth', 'corners']
        grouped_alarms, idxs_gids, df = cross_val.group_alarms_by(
            true_alarms, groupby_attrs=groupby_attrs,
            group_assign_attrs=groupby_attrs
        )

        self.assertEqual(len(true_alarms), len(df))
        self.assertEqual(len(true_alarms), sum(map(len, map(attrg('idxs'), grouped_alarms))))

        dfidxs_gids = dict(idxs_gids)
        gids_dfidxs = dict(grouped_alarms)
        for grouped_alarm in grouped_alarms:
            self.assertEqual(set(map(tuple, df[groupby_attrs].loc[grouped_alarm.idxs].values)),
                             {grouped_alarm.group_id})
            for idx in grouped_alarm.idxs:
                self.assertEqual(dfidxs_gids[idx], grouped_alarm.group_id)

        for groupd_alarm_index in idxs_gids:
            self.assertEqual(tuple(df.loc[groupd_alarm_index.idx][groupby_attrs].values),
                             groupd_alarm_index.group_id)
            self.assertTrue(groupd_alarm_index.idx in gids_dfidxs[groupd_alarm_index.group_id])

    def test_group_false_alarms(self):
        alarms = cross_val.tuf_table_csv_to_df(self.csv_fp)
        _, false_alarms = cross_val.split_t_f_alarms(alarms)
        grouped_alarms, idxs_gids, df = cross_val.group_false_alarms(false_alarms)

        self.assertEqual(len(false_alarms), len(df))
        self.assertEqual(len(false_alarms), sum(map(len, map(attrg('idxs'), grouped_alarms))))

        for f_alarm_grp in grouped_alarms:
            for ix in f_alarm_grp.idxs:
                for iy in f_alarm_grp.idxs:
                    self.assertLessEqual(
                        np.linalg.norm(np.asarray(df.loc[ix]['utm']) - np.asarray(df.loc[iy]['utm'])),
                        10 * np.sqrt(2)
                    )
