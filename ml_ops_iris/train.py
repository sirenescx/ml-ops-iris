from logging import Logger, getLogger
from pathlib import Path
from typing import Union

import pandas as pd

from ml_ops_iris.operations.common.load_dataset import DatasetLoadingOperation
from ml_ops_iris.operations.train.cross_validate import CrossValidationOperation
from ml_ops_iris.operations.train.preprocess_features import (
    FeaturesPreprocessingOperation,
)
from ml_ops_iris.operations.train.split_dataset import DatasetSplittingOperation
from ml_ops_iris.operations.train.train_model import ModelTrainingOperation
from ml_ops_iris.utils import get_directory


class TrainingPipeline:
    def __init__(
        self,
        dataset_loading_op: DatasetLoadingOperation,
        features_preprocessing_op: FeaturesPreprocessingOperation,
        dataset_splitting_op: DatasetSplittingOperation,
        model_training_op: ModelTrainingOperation,
        cross_validation_op: CrossValidationOperation,
    ):
        self._dataset_loading_op = dataset_loading_op
        self._features_preprocessing_op = features_preprocessing_op
        self._dataset_splitting_op = dataset_splitting_op
        self._model_training_op = model_training_op
        self._cross_validation_op = cross_validation_op

    def train(self, path: Union[Path, str], target_column_name: str = 'target'):
        logger: Logger = getLogger(__name__)

        logger.info('Started dataset preprocessing')
        data: pd.DataFrame = self._dataset_loading_op.load(path)
        features, target = self._dataset_splitting_op.split(
            data, target_column_name
        )
        features = self._features_preprocessing_op.preprocess(features)
        logger.info('Ended dataset preprocessing')

        logger.info('Started model training')
        model = self._model_training_op.train(
            features, target, get_directory(path)
        )
        logger.info('Ended model training')

        logger.info('Started cross validation')
        metrics: dict[str, float] = self._cross_validation_op.validate(
            model, features, target
        )
        logger.info('Ended cross validation')

        for metric_name, metric_value in metrics.items():
            logger.info('%s = %f', metric_name, metric_value)
