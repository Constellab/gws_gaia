# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import SGDRegressor

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# SGDRegressorResult
#
# *****************************************************************************


@resource_decorator("SGDRegressorResult", hide=True)
class SGDRegressorResult(BaseResource):
    pass

# *****************************************************************************
#
# SGDRegressorTrainer
#
# *****************************************************************************


@task_decorator("SGDRegressorTrainer", human_name="SGD regressor trainer",
                short_description="Train a stochastic gradient descent (SGD) linear regressor")
class SGDRegressorTrainer(Task):
    """
    Trainer of a linear regressor with stochastic gradient descent (SGD). Fit a SGD linear regressor with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(SGDRegressorResult, human_name="result", short_description="The output result")}
    config_specs = {
        'loss': StrParam(default_value='squared_loss'),
        'alpha': FloatParam(default_value=0.0001, min_value=0),
        'max_iter': IntParam(default_value=1000, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        sgdr = SGDRegressor(max_iter=params["max_iter"], alpha=params["alpha"], loss=params["loss"])
        sgdr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = SGDRegressorResult(training_set=dataset, result=sgdr)
        return {'result': result}

# *****************************************************************************
#
# SGDRegressorPredictor
#
# *****************************************************************************


@task_decorator("SGDRegressorPredictor", human_name="SGD regressor predictor",
                short_description="Predict targets using a trained SGD linear regressor")
class SGDRegressorPredictor(Task):
    """
    Predictor of a linear regressor with stochastic gradient descent (SGD). Predict target values of a dataset with a trained SGD linear regressor.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(SGDRegressorResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        sgdr = learned_model.get_result()
        y = sgdr.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
