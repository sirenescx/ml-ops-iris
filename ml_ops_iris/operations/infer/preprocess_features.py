from pathlib import Path
from typing import Union

import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler


class FeaturesPreprocessingOperation:
    def preprocess(
        self, features: pd.DataFrame, path: Union[Path, str] = 'std_scaler.bin'
    ) -> pd.DataFrame:
        scaler: StandardScaler = self._load_scaler(path)
        features = scaler.transform(features)

        return pd.DataFrame(features)

    def _load_scaler(self, path: Union[Path, str]) -> StandardScaler:
        try:
            scaler: StandardScaler = load(path)
            return scaler
        except Exception as exception:
            raise ValueError(
                f'Failed to load scaler, cause: {exception}'
            ) from exception
