# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from numpy import ravel
from pandas import DataFrame
from sklearn.svm import SVC

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam, BoolParam)

from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("SVCResult", hide=True)
class SVCResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("SVCTrainer")
class SVCTrainer(Task):
    """
    Trainer of a C-Support Vector Classifier (SVC) model. Fit a SVC model according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : SVCResult}
    config_specs = {
        'probability': BoolParam(default_value=False),
        'kernel':StrParam(default_value='rbf')
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        svc = SVC(probability=params["probability"],kernel=params["kernel"])
        svc.fit(dataset.features.values, ravel(dataset.targets.values))
        result = SVCResult.from_result(svc=svc)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("SVCTester")
class SVCTester(Task):
    """
    Tester of a trained C-Support Vector Classifier (SVC) model. Return the mean accuracy on a given dataset for a trained C-Support Vector Classifier (SVC) model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': SVCResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        svc = learned_model.binary_store['result']
        y = svc.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = GenericResult.from_result(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("SVCPredictor")
class SVCPredictor(Task):
    """
    Predictor of a C-Support Vector Classifier (SVC) model. Predict class labels of a dataset with a trained SVC model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html for more details.
    """    
    input_specs = {'dataset' : Dataset, 'learned_model': SVCResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        svc = learned_model.binary_store['result']
        y = svc.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}