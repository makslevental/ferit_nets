from functools import reduce
from itertools import product
from operator import concat
from typing import Union, Tuple, List, NamedTuple

import attr
import numpy as np
import pandas as pd

AlarmGroupId = Union[Tuple, int]


class GroupedAlarm(NamedTuple):
    group_id: AlarmGroupId
    idxs: pd.Index


class GroupedAlarmIndex(NamedTuple):
    idx: int
    group_id: AlarmGroupId


@attr.s(auto_attribs=True, frozen=True)
class GroupedAlarms(object):
    grouped_alarms: List[GroupedAlarm]
    df: pd.DataFrame

    def __attrs_post_init__(self):
        self.__dict__['_idxs_groupids'] = reduce(
            concat,
            [
                [GroupedAlarmIndex(idx, gid) for gid, idx in product([grouped_alarm.group_id], grouped_alarm.idxs)]
                for grouped_alarm in self.grouped_alarms
            ]
        )

    @property
    def idxs_groupids(self) -> List[GroupedAlarmIndex]:
        return self._idxs_groupids

    def __iter__(self):
        return iter([self.grouped_alarms, self.idxs_groupids, self.df])


class CrossValSplit(NamedTuple):
    nonholdout: pd.DataFrame
    holdout: pd.DataFrame


ROC = Tuple[np.ndarray, np.ndarray, np.ndarray]
