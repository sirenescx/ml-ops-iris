from dataclasses import dataclass

from ml_ops_iris.configs.base.storage_base import BaseStorageConfig


@dataclass
class OptimizerParametersConfig:
    iterations: int
    depth: int
    learning_rate: float
    loss_function: str


@dataclass
class ModelsConfig(BaseStorageConfig):
    optimizer_parameters: OptimizerParametersConfig
    custom_metrics: list[str]
