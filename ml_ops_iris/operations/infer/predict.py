from pathlib import Path

import pandas as pd
from sklearn.svm import SVC

from ml_ops_iris.utils import load_from_bin


class PredictionOperation:
    def predict(self, model_path: Path, features: pd.DataFrame):
        model: SVC = load_from_bin(path=model_path)
        return model.predict(features)
