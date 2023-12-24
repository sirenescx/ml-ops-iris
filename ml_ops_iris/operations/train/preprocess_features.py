from pathlib import Path
from typing import Union

import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler


class FeaturesPreprocessingOperation:
    def preprocess(
        self, features: pd.DataFrame, path: Union[Path, str] = 'std_scaler.bin'
    ) -> pd.DataFrame:
        scaler: StandardScaler = StandardScaler()
        features = scaler.fit_transform(features)
        dump(scaler, path)

        return pd.DataFrame(features)
