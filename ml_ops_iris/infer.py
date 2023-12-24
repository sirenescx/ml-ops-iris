from logging import Logger, getLogger
from pathlib import Path
from typing import Union

import pandas as pd

from ml_ops_iris.operations.common.load_dataset import DatasetLoadingOperation
from ml_ops_iris.operations.infer.load_model import ModelLoadingOperation
from ml_ops_iris.operations.infer.predict import PredictionOperation
from ml_ops_iris.operations.infer.preprocess_features import (
    FeaturesPreprocessingOperation,
)


class InferringPipeline:
    def __init__(
        self,
        dataset_loading_op: DatasetLoadingOperation,
        features_preprocessing_op: FeaturesPreprocessingOperation,
        model_loading_op: ModelLoadingOperation,
        prediction_op: PredictionOperation,
    ):
        self._dataset_loading_op = dataset_loading_op
        self._features_preprocessing_op = features_preprocessing_op
        self._model_loading_op = model_loading_op
        self._prediction_op = prediction_op

    def infer(self, path: Union[Path, str]):
        logger: Logger = getLogger(__name__)

        logger.info('Started dataset preprocessing')
        data: pd.DataFrame = self._dataset_loading_op.load(path)
        features: pd.DataFrame = self._features_preprocessing_op.preprocess(
            data
        )
        logger.info('Ended dataset preprocessing')

        logger.info('Started model loading')
        model = self._model_loading_op.load(path)
        logger.info('Ended model loading')

        logger.info('Started getting predictions')
        predictions = self._prediction_op.predict(model, features)
        logger.info('Ended getting predictions')

        logger.info(', '.join((str(prediction) for prediction in predictions)))
