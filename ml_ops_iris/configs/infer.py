from dataclasses import dataclass

from ml_ops_iris.configs.base.base import BaseConfig
from ml_ops_iris.configs.base.storage_base import BaseStorageConfig


@dataclass
class InferConfig(BaseConfig):
    dataset: BaseStorageConfig
    model: BaseStorageConfig
    predicts: BaseStorageConfig
