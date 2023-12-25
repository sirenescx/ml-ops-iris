from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ml_ops_iris.utils import construct_uri, get_latest_commit_id, save_to_bin


class ModelTrainingOperation:
    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_path: Path,
        optimizer_parameters,
        custom_metrics,
        ml_flow_parameters,
    ):
        mlflow.set_tracking_uri(
            uri=construct_uri(
                scheme=ml_flow_parameters.scheme,
                host=ml_flow_parameters.host,
                port=ml_flow_parameters.port,
            )
        )
        mlflow.set_experiment(str(int(datetime.now().timestamp())))

        trained_model: CatBoostClassifier = self._train_model(
            features=features,
            target=target,
            optimizer_parameters=optimizer_parameters,
            custom_metrics=custom_metrics,
        )
        save_to_bin(object=trained_model, path=model_path)

        return trained_model

    def _train_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        optimizer_parameters,
        custom_metrics,
    ):
        metrics = [optimizer_parameters.loss_function]
        metrics.extend(custom_metrics)

        model: CatBoostClassifier = CatBoostClassifier(
            **optimizer_parameters, custom_metric=metrics
        )

        class MlFlowCallback:
            def after_iteration(self, info):
                info = vars(info)
                for title, metric in (info.get('metrics') or {}).items():
                    for series, log in metric.items():
                        if series == optimizer_parameters.loss_function:
                            mlflow.log_metric('Loss', log[-1])
                        else:
                            mlflow.log_metric(series, log[-1])
                return True

        with mlflow.start_run():
            mlflow.log_param('code_version', get_latest_commit_id())
            mlflow.log_params(model.get_params())
            model.fit(Pool(features, target), callbacks=[MlFlowCallback()])
        mlflow.end_run()

        return model
