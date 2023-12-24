from pathlib import Path
from typing import Union

from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_from_bin(path: Path) -> Union[SVC, StandardScaler]:
    try:
        create_directories_if_not_exist(path=path)
        return load(filename=path)
    except Exception as exception:
        raise ValueError(
            f'Failed to load file, cause: {exception}'
        ) from exception


def save_to_bin(object: Union[SVC, StandardScaler], path: Path):
    try:
        create_directories_if_not_exist(path=path)
        dump(value=object, filename=path)
    except Exception as exception:
        raise ValueError(
            f'Failed to save file, cause: {exception}'
        ) from exception


def create_directories_if_not_exist(path: Path):
    path = path.absolute().parent
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
