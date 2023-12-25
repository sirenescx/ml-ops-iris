from pathlib import Path
from typing import Any
from urllib.parse import urlunparse

import git
from joblib import dump, load


def load_from_bin(path: Path) -> Any:
    try:
        create_directories_if_not_exist(path=path)
        return load(filename=path)
    except Exception as exception:
        raise ValueError(
            f'Failed to load file, cause: {exception}'
        ) from exception


def save_to_bin(object: Any, path: Path):
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


def construct_uri(scheme: str, host: str, port: str):
    return urlunparse((scheme, f'{host}:{port}', '/', '', '', ''))


def get_latest_commit_id():
    try:
        repo = git.Repo(search_parent_directories=True)
        commit_id = repo.head.commit.hexsha
        return commit_id
    except git.InvalidGitRepositoryError:
        print('Not a Git repository.')
        return None
