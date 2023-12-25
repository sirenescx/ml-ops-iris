from dataclasses import dataclass


@dataclass
class MlFlowConfig:
    scheme: str
    host: str
    port: str
