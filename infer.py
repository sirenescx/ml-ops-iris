from pathlib import Path

import hydra
from dvc.repo import Repo
from hydra.core.config_store import ConfigStore

from ml_ops_iris.configs.infer import InferConfig
from ml_ops_iris.infer import InferringPipeline
from ml_ops_iris.operations.common.load_dataset import DatasetLoadingOperation
from ml_ops_iris.operations.infer.predict import PredictionOperation
from ml_ops_iris.operations.infer.preprocess_features import (
    FeaturesPreprocessingOperation,
)
from ml_ops_iris.operations.infer.save_predictions import (
    PredictsSavingOperation,
)


CONFIG_STORE = ConfigStore.instance()
CONFIG_STORE.store(name='base_inference_config', node=InferConfig)


@hydra.main(version_base='1.2', config_path='configs', config_name='inference')
def main(config: InferConfig) -> None:
    data_directory: Path = Path(__file__).resolve().parent / 'data'

    repo = Repo('.dvc')
    repo.pull()

    pipeline: InferringPipeline = InferringPipeline(
        dataset_loading_op=DatasetLoadingOperation(),
        features_preprocessing_op=FeaturesPreprocessingOperation(),
        prediction_op=PredictionOperation(),
        predictions_saving_op=PredictsSavingOperation(),
    )
    pipeline.infer(
        dataset_path=data_directory / config.dataset.path,
        scaler_path=data_directory / config.scaler.path,
        model_path=data_directory / config.model.path,
        predicts_path=data_directory / config.predicts.path,
    )

    repo.add(targets=[data_directory.name])
    repo.commit()
    repo.push()


if __name__ == '__main__':
    main()
