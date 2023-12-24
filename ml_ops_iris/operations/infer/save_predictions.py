from pathlib import Path

import numpy as np
import pandas as pd

from ml_ops_iris.utils import create_directories_if_not_exist


class PredictsSavingOperation:
    def save(self, features: pd.DataFrame, predicts: np.ndarray, path: Path):
        features['predicts'] = predicts
        create_directories_if_not_exist(path=path)
        features.to_csv(path_or_buf=path, columns=features.columns.tolist())
