from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseStorageConfig:
    path: Path
