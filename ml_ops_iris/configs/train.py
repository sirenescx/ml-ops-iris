from dataclasses import dataclass

from ml_ops_iris.configs.base.base import BaseConfig
from ml_ops_iris.configs.datasets.train_dataset import TrainDatasetConfig
from ml_ops_iris.configs.ml_flow.ml_flow import MlFlowConfig
from ml_ops_iris.configs.models.models import ModelsConfig


@dataclass
class TrainConfig(BaseConfig):
    model: ModelsConfig
    dataset: TrainDatasetConfig
    ml_flow: MlFlowConfig
