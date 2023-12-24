from dataclasses import dataclass

from ml_ops_iris.configs.base.storage_base import BaseStorageConfig


@dataclass
class ParametersConfig:
    tolerance: float


@dataclass
class OptimizerParametersConfig:
    regularization: list[float]
    gamma: list[float]
    kernel: list[str]

    def as_dict(self):
        return self.as_dict()


@dataclass
class ModelsConfig(BaseStorageConfig):
    parameters: ParametersConfig
    optimizer_parameters: OptimizerParametersConfig
