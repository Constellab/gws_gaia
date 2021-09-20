# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import ExtraTreesClassifier

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("ExtraTreesClassifierResult", hide=True)
class ExtraTreesClassifierResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("ExtraTreesClassifierTrainer")
class ExtraTreesClassifierTrainer(Task):
    """
    Trainer of an extra-trees classifier. Build a forest of trees from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier.fit for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : ExtraTreesClassifierResult}
    config_specs = {
        'nb_estimators':IntParam(default_value=100, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        etc = ExtraTreesClassifier(n_estimators=params["nb_estimators"])
        etc.fit(dataset.features.values, ravel(dataset.targets.values))
        result = ExtraTreesClassifierResult(result = etc)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("ExtraTreesClassifierTester")
class ExtraTreesClassifierTester(Task):
    """
    Tester of a trained extra-trees classifier. Return the mean accuracy on a given dataset for a trained extra-trees classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': ExtraTreesClassifierResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        etc = learned_model.result
        y = etc.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = GenericResult(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("ExtraTreesClassifierPredictor")
class ExtraTreesClassifierPredictor(Task):
    """
    Predictor of an extra-trees classifier. Predict class for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier.fit for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': ExtraTreesClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        etc = learned_model.result
        y = etc.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}