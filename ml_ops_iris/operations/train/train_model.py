from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from ml_ops_iris.utils import save_to_bin


class ModelTrainingOperation:
    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_path: Path,
        parameters,
        optimizer_parameters,
    ):
        trained_model: SVC = self._optimize_hyperparameters(
            features=features,
            target=target,
            parameters=parameters,
            optimizer_parameters=optimizer_parameters,
        )
        save_to_bin(object=trained_model, path=model_path)
        return trained_model

    def _optimize_hyperparameters(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        parameters,
        optimizer_parameters,
    ):
        optimizer: GridSearchCV = GridSearchCV(
            estimator=SVC(tol=parameters.tolerance),
            param_grid={
                'kernel': optimizer_parameters.kernel,
                'C': optimizer_parameters.regularization,
                'gamma': optimizer_parameters.gamma,
            },
            verbose=3,
        )
        _ = optimizer.fit(features, target)
        return optimizer.best_estimator_
