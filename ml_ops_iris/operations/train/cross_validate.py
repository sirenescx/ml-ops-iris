from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC


class CrossValidationOperation:
    def validate(
        self, model: SVC, features: pd.DataFrame, target: pd.Series
    ) -> dict[str, float]:
        metrics: dict[str, float] = {
            'accuracy': self._cross_validate(
                model, features, target, accuracy_score
            ),
            'f1': self._cross_validate(
                model, features, target, partial(f1_score, average='weighted')
            ),
            'precision': self._cross_validate(
                model,
                features,
                target,
                partial(precision_score, average='weighted'),
            ),
            'recall': self._cross_validate(
                model,
                features,
                target,
                partial(recall_score, average='weighted'),
            ),
        }
        return metrics

    def _cross_validate(
        self, model: SVC, features: pd.DataFrame, target: pd.Series, metric
    ):
        leave_one_out: LeaveOneOut = LeaveOneOut()
        predicted_target = np.zeros(target.shape)

        for train_index, test_index in leave_one_out.split(features):
            train_features, eval_features = (
                features.values[train_index],
                features.values[test_index],
            )
            train_target = target.values[train_index]
            model.fit(train_features, train_target)
            predicted_target[test_index] = model.predict(eval_features)
        mean_metric_score = metric(predicted_target, target)

        return mean_metric_score
