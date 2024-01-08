from pathlib import Path

import hydra
from dvc.repo import Repo
from hydra.core.config_store import ConfigStore

from ml_ops_iris.configs.train_config import TrainConfig
from ml_ops_iris.operations.common.load_dataset import DatasetLoadingOperation
from ml_ops_iris.operations.train.preprocess_features import (
    FeaturesPreprocessingOperation,
)
from ml_ops_iris.operations.train.split_dataset import DatasetSplittingOperation
from ml_ops_iris.operations.train.train_model import ModelTrainingOperation
from ml_ops_iris.train_pipeline import TrainingPipeline


CONFIG_STORE = ConfigStore.instance()
CONFIG_STORE.store(name='base_train_config', node=TrainConfig)


@hydra.main(version_base='1.2', config_path='configs', config_name='train')
def main(config: TrainConfig) -> None:
    data_directory: Path = Path(__file__).resolve().parent / 'data'

    repo = Repo('.dvc')
    repo.pull()

    pipeline: TrainingPipeline = TrainingPipeline(
        dataset_loading_op=DatasetLoadingOperation(),
        dataset_splitting_op=DatasetSplittingOperation(),
        features_preprocessing_op=FeaturesPreprocessingOperation(),
        model_training_op=ModelTrainingOperation(),
    )
    pipeline.train(
        dataset_path=data_directory / config.dataset.path,
        scaler_path=data_directory / config.scaler.path,
        model_path=data_directory / config.model.path,
        target_column=config.dataset.target_column,
        optimizer_parameters=config.model.optimizer_parameters,
        custom_metrics=config.model.custom_metrics,
        ml_flow_parameters=config.ml_flow,
    )

    repo.add(targets=[data_directory.name])
    repo.commit()
    repo.push()


if __name__ == '__main__':
    main()
