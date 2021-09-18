# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("KNNClassifierResult", hide=True)
class KNNClassifierResult(Resource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("KNNClassifierTrainer")
class KNNClassifierTrainer(Task):
    """
    Trainer of a k-nearest neighbors classifier. Fit the k-nearest neighbors classifier from the training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : KNNClassifierResult}
    config_specs = {
        'nb_neighbors': IntParam(default_value=5, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        neigh = KNeighborsClassifier(n_neighbors=params["nb_neighbors"])
        neigh.fit(dataset.features.values, ravel(dataset.targets.values))
        result = KNNClassifierResult.from_result(result=neigh)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("KNNClassifierTester")
class KNNClassifierTester(Task):
    """
    Tester of a trained K-nearest neighbors classifier. Return the mean accuracy on a given dataset for a trained K-nearest neighbors classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': KNNClassifierResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        neigh = learned_model.binary_store['result']
        y = neigh.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = GenericResult.from_result(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("KNNClassifierPredictor")
class KNNClassifierPredictor(Task):
    """
    Predictor of a K-nearest neighbors classifier. Predict the class labels for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': KNNClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        neigh = learned_model.binary_store['result']
        y = neigh.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}