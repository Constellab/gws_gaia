# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.gaussian_process import GaussianProcessRegressor

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("GaussianProcessRegressorResult", hide=True)
class GaussianProcessRegressorResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("GaussianProcessRegressorTrainer")
class GaussianProcessRegressorTrainer(Task):
    """
    Trainer of a Gaussian process regressor. Fit a Gaussian process regressor model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html for more details
    """    
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : GaussianProcessRegressorResult}
    config_specs = {
        'alpha': FloatParam(default_value=1e-10, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        gpr = GaussianProcessRegressor(alpha=params["alpha"])
        gpr.fit(dataset.features.values, ravel(dataset.targets.values))
        result = GaussianProcessRegressorResult(result=gpr)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("GaussianProcessRegressorTester")
class GaussianProcessRegressorTester(Task):
    """
    Tester of a trained Gaussian process regressor. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained Gaussian process regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GaussianProcessRegressorResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gpr = learned_model.result
        y = gpr.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = GenericResult(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("GaussianProcessRegressorPredictor")
class GaussianProcessRegressorPredictor(Task):
    """
    Predictor of a Gaussian process regressor. Predict regression targets of a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html for more details
    """    
    input_specs = {'dataset' : Dataset, 'learned_model': GaussianProcessRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gpr = learned_model.result
        y = gpr.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}