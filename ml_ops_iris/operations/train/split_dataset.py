import pandas as pd


class DatasetSplittingOperation:
    def split(self, dataset: pd.DataFrame, target_column_name: str):
        dataset = dataset.drop_duplicates()
        features = dataset.drop(target_column_name, axis=1)
        target = dataset[target_column_name]

        return features, target
