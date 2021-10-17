# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.gaussian_process import GaussianProcessClassifier

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("GaussianProcessClassifierResult", hide=True)
class GaussianProcessClassifierResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("GaussianProcessClassifierTrainer")
class GaussianProcessClassifierTrainer(Task):
    """
    Trainer of a Gaussian process classifier. Fit a Gaussian process classification model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier.fit for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : GaussianProcessClassifierResult}
    config_specs = {
        'random_state':IntParam(default_value=None, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        gpc = GaussianProcessClassifier(random_state=params["random_state"])
        gpc.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = GaussianProcessClassifierResult(result = gpc)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("GaussianProcessClassifierTester")
class GaussianProcessClassifierTester(Task):
    """
    Tester of a trained Gaussian process classifier. Return the mean accuracy on a given dataset for a trained Gaussian process classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GaussianProcessClassifierResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gpc = learned_model.result
        y = gpc.score(dataset.get_features().values, dataset.get_targets().values)
        z = tuple([y])
        result_dataset = GenericResult(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("GaussianProcessClassifierPredictor")
class GaussianProcessClassifierPredictor(Task):
    """
    Predictor of a Gaussian process classifier. Predict classes of a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier.fit for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GaussianProcessClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gpc = learned_model.result
        y = gpc.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}