from pathlib import Path
from typing import Union


def exists(filepath: Union[Path, str]) -> bool:
    return cast_to_path(filepath).exists()


def is_file(filepath: Union[Path, str]) -> bool:
    return cast_to_path(filepath).is_file()


def get_directory(filepath: Union[Path, str]) -> Path:
    return cast_to_path(filepath).parent.absolute()


def get_filename(filepath: str) -> str:
    return cast_to_path(filepath).name


def get_filepath(directory: Union[Path, str], filename: Union[Path, str]):
    return get_directory(cast_to_path(directory)) / cast_to_path(filename)


def cast_to_path(filepath: Union[Path, str]) -> Path:
    if isinstance(filepath, str):
        filepath = Path(filepath)
    return filepath
