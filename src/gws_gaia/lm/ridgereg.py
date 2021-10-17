# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import Ridge

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("RidgeRegressionResult", hide=True)
class RidgeRegressionResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("RidgeRegressionTrainer")
class RidgeRegressionTrainer(Task):
    """
    Trainer of a Ridge regression model. Fit a Ridge regression model with a training dataset. 

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : RidgeRegressionResult}
    config_specs = {
        'alpha':FloatParam(default_value=1, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        rir = Ridge(alpha=params["alpha"])
        rir.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = RidgeRegressionResult(result = rir)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("RidgeRegressionTester")
class RidgeRegressionTester(Task):
    """
    Tester of a trained Ridge regression model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained Ridge regression model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': RidgeRegressionResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        rir = learned_model.result
        y = rir.score(dataset.get_features().values, dataset.get_targets().values)
        z = tuple([y])
        result_dataset = GenericResult(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("RidgeRegressionPredictor")
class RidgeRegressionPredictor(Task):
    """
    Predictor of a Ridge regression model. Predict target values of a dataset with a trained Ridge regression model. 

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': RidgeRegressionResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        rir = learned_model.result
        y = rir.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}