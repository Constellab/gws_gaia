# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.kernel_ridge import KernelRidge

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("KernelRidgeResult", hide=True)
class KernelRidgeResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("KernelRidgeTrainer")
class KernelRidgeTrainer(Task):
    """
    Trainer of a kernel ridge regression model. Fit a kernel ridge regression model with a training dataset. 

    See https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : KernelRidgeResult}
    config_specs = {
        'gamma': FloatParam(default_value=None),
        'kernel': StrParam(default_value='linear')
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        krr = KernelRidge(gamma=params["gamma"],kernel=params["kernel"])
        krr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = KernelRidgeResult(result = krr)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("KernelRidgeTester")
class KernelRidgeTester(Task):
    """
    Tester of a trained kernel ridge regression model. Return the coefficient of determination R^2 of the prediction on a given dataset for a kernel ridge regression model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': KernelRidgeResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        krr = learned_model.result
        y = krr.score(dataset.get_features().values, dataset.get_targets().values)
        z = tuple([y])
        result_dataset = GenericResult(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("KernelRidgePredictor")
class KernelRidgePredictor(Task):
    """
    Predictor of a kernel ridge regression model. Predict a regression target from a dataset with a trained kernel ridge regression model. 

    See https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': KernelRidgeResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        krr = learned_model.result
        y = krr.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}