from pathlib import Path

import pandas as pd


class DatasetLoadingOperation:
    def load(self, path: Path):
        if not path.exists():
            raise ValueError(
                f'Failed to load dataset: path {path} does not exist'
            )
        if not path.is_file():
            raise ValueError(f'Failed to load dataset: {path} is not a file')
        try:
            dataset: pd.DataFrame = pd.read_csv(path)
            return dataset
        except Exception as exception:
            raise ValueError(
                f'Failed to load dataset, cause: {exception}'
            ) from exception
