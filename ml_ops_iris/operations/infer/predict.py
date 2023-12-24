import pandas as pd
from sklearn.svm import SVC


class PredictionOperation:
    def predict(self, model: SVC, features: pd.DataFrame):
        return model.predict(features)
