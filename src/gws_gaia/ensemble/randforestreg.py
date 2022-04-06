# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, IntParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from numpy import ravel
from sklearn.ensemble import RandomForestRegressor

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# RandomForestRegressorResult
#
# *****************************************************************************


@resource_decorator("RandomForestRegressorResult", hide=True)
class RandomForestRegressorResult(BaseResource):
    pass

# *****************************************************************************
#
# RandomForestRegressorTrainer
#
# *****************************************************************************


@task_decorator("RandomForestRegressorTrainer", human_name="Random-Forest regressor trainer",
                short_description="Train a random forest regression model")
class RandomForestRegressorTrainer(Task):
    """
    Trainer of a random forest regressor. Build a forest of trees from a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html for more details.
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': RandomForestRegressorResult}
    config_specs = {
        'nb_estimators': IntParam(default_value=100, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        rfr = RandomForestRegressor(n_estimators=params["nb_estimators"])
        rfr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = RandomForestRegressorResult(training_set=dataset, result=rfr)
        return {'result': result}

# *****************************************************************************
#
# RandomForestRegressorPredictor
#
# *****************************************************************************


@task_decorator("RandomForestRegressorPredictor", human_name="Random-Forest regression predictor",
                short_description="Predict dataset targets using a trained Random forest regression model")
class RandomForestRegressorPredictor(Task):
    """
    Predictor of a random forest regressor. Predict regression target of a dataset with a trained random forest regressor.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html for more details.
    """
    input_specs = {'dataset': Dataset, 'learned_model': RandomForestRegressorResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        rfr = learned_model.get_result()
        y = rfr.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
