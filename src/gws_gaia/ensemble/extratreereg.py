# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import ExtraTreesRegressor

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.core import Tuple
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("ExtraTreesRegressorResult", hide=True)
class ExtraTreesRegressorResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("ExtraTreesRegressorTrainer")
class ExtraTreesRegressorTrainer(Task):
    """
    Trainer of an extra-trees regressor. Build a forest of trees from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : ExtraTreesRegressorResult}
    config_specs = {
        'nb_estimators':IntParam(default_value=100, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        etr = ExtraTreesRegressor(n_estimators=params["nb_estimators"])
        etr.fit(dataset.features.values, ravel(dataset.targets.values))
        result = ExtraTreesRegressorResult.from_result(result=etr)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("ExtraTreesRegressorTester")
class ExtraTreesRegressorTester(Task):
    """
    Tester of a trained extra-trees regressor. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained extra-trees regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': ExtraTreesRegressorResult}
    output_specs = {'result' : Tuple}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        etr = learned_model.binary_store['result']
        y = etr.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = Tuple(tup = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("ExtraTreesRegressorPredictor")
class ExtraTreesRegressorPredictor(Task):
    """
    Predictor of an extra-trees regressor. Predict regression target of a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': ExtraTreesRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        etr = learned_model.binary_store['result']
        y = etr.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}