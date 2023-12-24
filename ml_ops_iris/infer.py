from logging import Logger, getLogger
from pathlib import Path

import pandas as pd

from ml_ops_iris.operations.common.load_dataset import DatasetLoadingOperation
from ml_ops_iris.operations.infer.predict import PredictionOperation
from ml_ops_iris.operations.infer.preprocess_features import (
    FeaturesPreprocessingOperation,
)


class InferringPipeline:
    def __init__(
        self,
        dataset_loading_op: DatasetLoadingOperation,
        features_preprocessing_op: FeaturesPreprocessingOperation,
        prediction_op: PredictionOperation,
    ):
        self._dataset_loading_op = dataset_loading_op
        self._features_preprocessing_op = features_preprocessing_op
        self._prediction_op = prediction_op

    def infer(
        self,
        dataset_path: Path,
        scaler_path: Path,
        model_path: Path,
        predicts_path: Path,
    ):
        logger: Logger = getLogger(__name__)

        logger.info('Started dataset preprocessing')
        data: pd.DataFrame = self._dataset_loading_op.load(path=dataset_path)
        features: pd.DataFrame = self._features_preprocessing_op.preprocess(
            features=data, scaler_path=scaler_path
        )
        logger.info('Ended dataset preprocessing')

        logger.info('Started getting predictions')
        predictions = self._prediction_op.predict(
            model_path=model_path, features=features
        )
        logger.info('Ended getting predictions')

        logger.info(', '.join((str(prediction) for prediction in predictions)))
