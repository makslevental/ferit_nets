from functools import reduce
from itertools import product
from operator import concat
from typing import Union, Tuple, List, NamedTuple

import numpy as np
import pandas as pd

AlarmGroupId = Union[Tuple, int]


class GroupedAlarm(NamedTuple):
    group_id: AlarmGroupId
    idxs: pd.Index


class GroupedAlarmIndex(NamedTuple):
    idx: int
    group_id: AlarmGroupId


class GroupedAlarms(NamedTuple):
    grouped_alarms: List[GroupedAlarm]
    df: pd.DataFrame

    @property
    def idxs_groupids(self) -> List[GroupedAlarmIndex]:
        return reduce(
            concat,
            [
                [GroupedAlarmIndex(idx, gid) for gid, idx in product([grouped_alarm.group_id], grouped_alarm.idxs)]
                for grouped_alarm in self.grouped_alarms
            ]
        )


class CrossValSplit(NamedTuple):
    nonholdout: pd.DataFrame
    holdout: pd.DataFrame


ROC = Tuple[np.ndarray, np.ndarray, np.ndarray]
