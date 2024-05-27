![python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) ![dvc](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white) ![catboost](https://img.shields.io/badge/catboost-ffc300?style=for-the-badge&logo=catboost) ![mlflow](https://img.shields.io/badge/mlflow-white?style=for-the-badge&logo=mlflow)

This repository contains a solution to the multiclass classification problem on the Iris
dataset using the CatBoostClassifier model.

### How to use

#### Basic usage

-   python3 train.py for model training
-   python3 infer.py to get predictions on the test dataset, dataset with obtained
    predictions will be stored in data > predicts

#### Advanced usage

##### Modifying training hyperparameters

Go to file train.yaml located in the "configs" directory and modify model >
optimizer_parameters section

##### Modifying metrics

Go to file train.yaml located in the "configs" directory and modify model > custom metrics
section (please refer to
[CatBoost documentation](https://catboost.ai/en/docs/concepts/loss-functions-multiclassification)
to get available metrics)

### Implementation details

#### Data

-   Datasets are stored in dvc using Google Drive as backend
-   All files created during training / inferring are also saved to dvc

#### Logging

-   Training parameters and metrics are logged using MLFlow
-   Start and end of training / inferring steps are logged to console
