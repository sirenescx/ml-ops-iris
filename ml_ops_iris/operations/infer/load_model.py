import pickle
from pathlib import Path
from typing import Union

from ml_ops_iris.utils import get_filepath


class ModelLoadingOperation:
    def load(self, directory: Union[Path, str], filename: str = 'SVC.pickle'):
        model_path: str = get_filepath(directory=directory, filename=filename)
        with open(model_path, 'rb') as io:
            loaded_model = pickle.load(io)

        return loaded_model
