# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.neighbors import KNeighborsRegressor

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.core import Tuple
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("KNNRegressorResult", hide=True)
class KNNRegressorResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("KNNRegressorTrainer")
class KNNRegressorTrainer(Task):
    """
    Trainer for a k-nearest neighbors regressor. Fit a k-nearest neighbors regressor from a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : KNNRegressorResult}
    config_specs = {
        'nb_neighbors': IntParam(default_value=5, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        neigh = KNeighborsRegressor(n_neighbors=params["nb_neighbors"])
        neigh.fit(dataset.features.values, ravel(dataset.targets.values))
        result = KNNRegressorResult.from_result(result=neigh)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("KNNRegressorTester")
class KNNRegressorTester(Task):
    """
    Tester of a trained k-nearest neighbors regressor. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained k-nearest neighbors regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': KNNRegressorResult}
    output_specs = {'result' : Tuple}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        neigh = learned_model.binary_store['result']
        y = neigh.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = Tuple(tup = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("KNNRegressorPredictor")
class KNNRegressorPredictor(Task):
    """
    Predictor for a k-nearest neighbors regressor. Predict the regression target for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': KNNRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        neigh = learned_model.binary_store['result']
        y = neigh.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}