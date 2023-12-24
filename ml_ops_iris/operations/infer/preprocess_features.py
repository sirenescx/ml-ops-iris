from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_ops_iris.utils import load_from_bin


class FeaturesPreprocessingOperation:
    def preprocess(
        self, features: pd.DataFrame, scaler_path: Path
    ) -> pd.DataFrame:
        feature_names: list[str] = features.columns.tolist()
        scaler: StandardScaler = load_from_bin(path=scaler_path)
        features = scaler.transform(features)
        return pd.DataFrame(features, columns=feature_names)
