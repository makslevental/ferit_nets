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
    dfidxs_groupids: List[GroupedAlarmIndex]
    df: pd.DataFrame


class CrossValSplit(NamedTuple):
    nonholdout: pd.DataFrame
    holdout: pd.DataFrame


ROC = Tuple[np.ndarray, np.ndarray, np.ndarray]
