# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from sklearn.gaussian_process import GaussianProcessRegressor

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# GaussianProcessRegressorResult
#
# *****************************************************************************


@resource_decorator("GaussianProcessRegressorResult", hide=True)
class GaussianProcessRegressorResult(BaseResource):
    pass

# *****************************************************************************
#
# GaussianProcessRegressorTrainer
#
# *****************************************************************************


@task_decorator("GaussianProcessRegressorTrainer", human_name="Gaussian process regressor trainer",
                short_description="Train a Gaussian process regression model")
class GaussianProcessRegressorTrainer(Task):
    """
    Trainer of a Gaussian process regressor. Fit a Gaussian process regressor model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html for more details
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(GaussianProcessRegressorResult, human_name="result", short_description="The output result")}
    config_specs = {
        'alpha': FloatParam(default_value=1e-10, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        gpr = GaussianProcessRegressor(alpha=params["alpha"])
        gpr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = GaussianProcessRegressorResult(training_set=dataset, result=gpr)
        return {'result': result}

# *****************************************************************************
#
# GaussianProcessRegressorPredictor
#
# *****************************************************************************


@task_decorator("GaussianProcessRegressorPredictor", human_name="Gaussian process regressor predictor",
                short_description="Predict dataset targets using a trained Gaussian process regression model")
class GaussianProcessRegressorPredictor(Task):
    """
    Predictor of a Gaussian process regressor. Predict regression targets of a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html for more details
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(GaussianProcessRegressorResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gpr = learned_model.get_result()
        y = gpr.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
