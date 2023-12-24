import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from ml_ops_iris.utils import get_filepath


class ModelTrainingOperation:
    def train(self, features: pd.DataFrame, target: pd.Series, directory: Path):
        trained_model: SVC = self._optimize_hyperparameters(features, target)
        self._save_model(trained_model, directory)
        return trained_model

    def _optimize_hyperparameters(
        self, features: pd.DataFrame, target: pd.Series
    ):
        optimizer: GridSearchCV = GridSearchCV(
            estimator=SVC(),
            param_grid={
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf'],
            },
            verbose=3,
        )
        _ = optimizer.fit(features, target)
        return optimizer.best_estimator_

    def _save_model(self, model: SVC, directory: Path, filename='SVC.pickle'):
        model_path: str = get_filepath(directory=directory, filename=filename)
        with open(model_path, 'wb') as io:
            pickle.dump(model, io)
