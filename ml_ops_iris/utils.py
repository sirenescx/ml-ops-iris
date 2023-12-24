from pathlib import Path
from typing import Union

from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_from_bin(path: Path) -> Union[SVC, StandardScaler]:
    try:
        return load(path)
    except Exception as exception:
        raise ValueError(
            f'Failed to load file, cause: {exception}'
        ) from exception


def save_to_bin(object: Union[SVC, StandardScaler], path: Path):
    try:
        dump(object, path)
    except Exception as exception:
        raise ValueError(
            f'Failed to save file, cause: {exception}'
        ) from exception
