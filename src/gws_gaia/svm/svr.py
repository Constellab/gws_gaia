# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.svm import SVR

from gws_core import (Task, Resource, resource_decorator, task_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("SVRResult", hide=True)
class SVRResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("SVRTrainer")
class SVRTrainer(Task):
    """
    Trainer of a Epsilon-Support Vector Regression (SVR) model. Fit a SVR model according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : SVRResult}
    config_specs = {
        'kernel':StrParam(default_value='rbf')
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        svr = SVR(kernel=params["kernel"])
        svr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = SVRResult(result=svr)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("SVRTester")
class SVRTester(Task):
    """
    Tester of a trained Epsilon-Support Vector Regression (SVR) model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained Epsilon-Support Vector Regression (SVR) model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': SVRResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        svr = learned_model.result
        y = svr.score(dataset.get_features().values, dataset.get_targets().values)
        z = tuple([y])
        result_dataset = GenericResult(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("SVRPredictor")
class SVRPredictor(Task):
    """
    Predictor of a Epsilon-Support Vector Regression (SVR) model. Predict target values of a dataset with a trained SVR model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html for more details.
    """    
    input_specs = {'dataset' : Dataset, 'learned_model': SVRResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        svr = learned_model.result
        y = svr.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}