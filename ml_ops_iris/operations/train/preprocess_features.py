from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_ops_iris.utils import save_to_bin


class FeaturesPreprocessingOperation:
    def preprocess(
        self, features: pd.DataFrame, scaler_path: Path
    ) -> pd.DataFrame:
        feature_names: list[str] = features.columns.tolist()
        scaler: StandardScaler = StandardScaler()
        features = scaler.fit_transform(features)
        save_to_bin(object=scaler, path=scaler_path)

        return pd.DataFrame(features, columns=feature_names)
