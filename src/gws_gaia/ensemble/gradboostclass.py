# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.core import Tuple
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("GradientBoostingClassifierResult", hide=True)
class GradientBoostingClassifierResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("GradientBoostingClassifierTrainer")
class GradientBoostingClassifierTrainer(Task):
    """
    Trainer of a gradient boosting classifier. Fit a gradient boosting classifier with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : GradientBoostingClassifierResult}
    config_specs = {
        'nb_estimators':IntParam(default_value=100, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        gbc = GradientBoostingClassifier(n_estimators=params["nb_estimators"])
        gbc.fit(dataset.features.values, ravel(dataset.targets.values))
        result = GradientBoostingClassifierResult.from_result(result=gbc)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("GradientBoostingClassifierTester")
class GradientBoostingClassifierTester(Task):
    """
    Tester of a trained gradient boosting classifier. Return the mean accuracy on a given dataset for a trained gradient boosting classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GradientBoostingClassifierResult}
    output_specs = {'result' : Tuple}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gbc = learned_model.binary_store['result']
        y = gbc.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = Tuple(tup = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("GradientBoostingClassifierPredictor")
class GradientBoostingClassifierPredictor(Task):
    """
    Predictor of a gradient boosting classifier. Predict classes for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GradientBoostingClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gbc = learned_model.binary_store['result']
        y = gbc.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}