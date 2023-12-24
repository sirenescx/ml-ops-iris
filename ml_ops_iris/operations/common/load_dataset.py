from pathlib import Path
from typing import Union

import pandas as pd

from ml_ops_iris.utils import exists, is_file


class DatasetLoadingOperation:
    def load(self, path: Union[Path, str]):
        if not exists(path):
            raise ValueError(
                f'Failed to load dataset: path {path} does not exist'
            )
        if not is_file(path):
            raise ValueError(f'Failed to load dataset: {path} is not a file')
        try:
            dataset: pd.DataFrame = pd.read_csv(path)
            return dataset
        except Exception as exception:
            raise ValueError(
                f'Failed to load dataset, cause: {exception}'
            ) from exception
