from dataclasses import dataclass

from ml_ops_iris.configs.base.storage_base import BaseStorageConfig


@dataclass
class BaseConfig:
    scaler: BaseStorageConfig
