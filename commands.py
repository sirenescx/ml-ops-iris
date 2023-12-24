"""
Module providing methods for training and inferring SVC model on iris dataset
"""
import hydra
from omegaconf import DictConfig

from ml_ops_iris.infer import InferringPipeline
from ml_ops_iris.operations.common.load_dataset import DatasetLoadingOperation
from ml_ops_iris.operations.infer.load_model import ModelLoadingOperation
from ml_ops_iris.operations.infer.predict import PredictionOperation
from ml_ops_iris.operations.train.cross_validate import CrossValidationOperation
from ml_ops_iris.operations.train.preprocess_features import (
    FeaturesPreprocessingOperation as TrainFeaturesPreprocessingOperation,
)
from ml_ops_iris.operations.train.split_dataset import DatasetSplittingOperation
from ml_ops_iris.operations.train.train_model import ModelTrainingOperation
from ml_ops_iris.train import TrainingPipeline


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(
    config: DictConfig,
):
    launch_mode: str = config['mode']
    if launch_mode == 'train':
        pipeline: TrainingPipeline = TrainingPipeline(
            dataset_loading_op=DatasetLoadingOperation(),
            dataset_splitting_op=DatasetSplittingOperation(),
            features_preprocessing_op=TrainFeaturesPreprocessingOperation(),
            model_training_op=ModelTrainingOperation(),
            cross_validation_op=CrossValidationOperation(),
        )
        pipeline.train('data/iris_train.csv')
    elif launch_mode == 'infer':
        pipeline: InferringPipeline = InferringPipeline(
            dataset_loading_op=DatasetLoadingOperation(),
            features_preprocessing_op=TrainFeaturesPreprocessingOperation(),
            model_loading_op=ModelLoadingOperation(),
            prediction_op=PredictionOperation(),
        )
        pipeline.infer('data/iris_test.csv')


if __name__ == '__main__':
    main()
