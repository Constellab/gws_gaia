# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import RidgeClassifier

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import Tuple
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("RidgeClassifierResult", hide=True)
class RidgeClassifierResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("RidgeClassifierTrainer")
class RidgeClassifierTrainer(Task):
    """
    Trainer of a Ridge regression classifier. Fit a Ridge classifier model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : RidgeClassifierResult}
    config_specs = {
        'alpha':FloatParam(default_value=1, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        ric = RidgeClassifier(alpha=params["alpha"])
        ric.fit(dataset.features.values, ravel(dataset.targets.values))
        result = RidgeClassifierResult.from_result(result=ric)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("RidgeClassifierTester")
class RidgeClassifierTester(Task):
    """
    Tester of a trained Ridge regression classifier. Return the mean accuracy on a given dataset for a trained Ridge regression classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': RidgeClassifierResult}
    output_specs = {'result' : Tuple}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        ric = learned_model.binary_store['result']
        y = ric.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = Tuple(tup = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("RidgeClassifierPredictor")
class RidgeClassifierPredictor(Task):
    """
    Predictor of a Ridge regression classifier. Predict class labels for samples in a datatset with a trained Ridge classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for more details.
    """    
    input_specs = {'dataset' : Dataset, 'learned_model': RidgeClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        ric = learned_model.binary_store['result']
        y = ric.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}