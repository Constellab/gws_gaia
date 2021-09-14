# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.core import Tuple
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("RandomForestClassifierResult", hide=True)
class RandomForestClassifierResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("RandomForestClassifierTrainer")
class RandomForestClassifierTrainer(Task):
    """
    Trainer of a random forest classifier. Build a forest of trees from a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : RandomForestClassifierResult}
    config_specs = {
        'nb_estimators':IntParam(default_value=100, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        rfc = RandomForestClassifier(n_estimators=params["nb_estimators"])
        rfc.fit(dataset.features.values, ravel(dataset.targets.values))
        result = RandomForestClassifierResult.from_result(result=rfc)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("RandomForestClassifierTester")
class RandomForestClassifierTester(Task):
    """
    Tester of a trained random forest classifier. Return the mean accuracy on a given dataset for a trained random forest classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': RandomForestClassifierResult}
    output_specs = {'result' : Tuple}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        rfc = learned_model.binary_store['result']
        y = rfc.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = Tuple(tup = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("RandomForestClassifierPredictor")
class RandomForestClassifierPredictor(Task):
    """
    Predictor of a random forest classifier. Predict class labels of a dataset with a trained random forest classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': RandomForestClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        rfc = learned_model.binary_store['result']
        y = rfc.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}