# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import SGDRegressor

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.dataset import Dataset
from ..data.core import GenericResult
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("SGDRegressorResult", hide=True)
class SGDRegressorResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("SGDRegressorTrainer")
class SGDRegressorTrainer(Task):
    """
    Trainer of a linear regressor with stochastic gradient descent (SGD). Fit a SGD linear regressor with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for more details.
    """    
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : SGDRegressorResult}
    config_specs = {
        'loss':StrParam(default_value='squared_loss'),
        'alpha': FloatParam(default_value=0.0001, min_value=0),
        'max_iter':IntParam(default_value=1000, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        sgdr = SGDRegressor(max_iter=params["max_iter"],alpha=params["alpha"],loss=params["loss"])
        sgdr.fit(dataset.features.values, ravel(dataset.targets.values))
        result = SGDRegressorResult.from_result(result=sgdr)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("SGDRegressorTester")
class SGDRegressorTester(Task):
    """
    Tester of a trained linear regressor with stochastic gradient descent (SGD). Return the coefficient of determination R^2 of the prediction on a given dataset for a trained linear regressor with stochastic gradient descent (SGD).
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': SGDRegressorResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        sgdr = learned_model.binary_store['result']
        y = sgdr.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = GenericResult.from_result(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("SGDRegressorPredictor")
class SGDRegressorPredictor(Task):
    """
    Predictor of a linear regressor with stochastic gradient descent (SGD). Predict target values of a dataset with a trained SGD linear regressor.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for more details.
    """        
    input_specs = {'dataset' : Dataset, 'learned_model': SGDRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        sgdr = learned_model.binary_store['result']
        y = sgdr.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}