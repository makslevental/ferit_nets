from typing import Union, Tuple, List

import numpy as np
import pandas as pd

AlarmGroupId = Union[Tuple, int]
GroupedAlarms = Tuple[AlarmGroupId, pd.Int64Index]
AlarmGroups = Tuple[List[GroupedAlarms], List[Tuple[int, AlarmGroupId]]]

CrossValSplit = Tuple[pd.DataFrame, pd.DataFrame]
ROC = Tuple[np.ndarray, np.ndarray, np.ndarray]
