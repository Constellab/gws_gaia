# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator)
from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import Ridge

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# RidgeRegressionResult
#
# *****************************************************************************


@resource_decorator("RidgeRegressionResult", hide=True)
class RidgeRegressionResult(BaseResource):
    pass

# *****************************************************************************
#
# RidgeRegressionTrainer
#
# *****************************************************************************


@task_decorator("RidgeRegressionTrainer", human_name="Ridge regression trainer",
                short_description="Train a ridge regression model")
class RidgeRegressionTrainer(Task):
    """
    Trainer of a Ridge regression model. Fit a Ridge regression model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more details.
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': RidgeRegressionResult}
    config_specs = {
        'alpha': FloatParam(default_value=1, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        rir = Ridge(alpha=params["alpha"])
        rir.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = RidgeRegressionResult(training_set=dataset, result=rir)
        return {'result': result}

# *****************************************************************************
#
# RidgeRegressionPredictor
#
# *****************************************************************************


@task_decorator("RidgeRegressionPredictor", human_name="Ridge regression predictor",
                short_description="Predict dataset targets using a trained ridge regression model")
class RidgeRegressionPredictor(Task):
    """
    Predictor of a Ridge regression model. Predict target values of a dataset with a trained Ridge regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more details.
    """
    input_specs = {'dataset': Dataset, 'learned_model': RidgeRegressionResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        rir = learned_model.get_result()
        y = rir.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
