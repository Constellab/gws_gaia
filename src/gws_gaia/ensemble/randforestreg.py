# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from gws_core import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("RandomForestRegressorResult", hide=True)
class RandomForestRegressorResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("RandomForestRegressorTrainer")
class RandomForestRegressorTrainer(Task):
    """
    Trainer of a random forest regressor. Build a forest of trees from a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : RandomForestRegressorResult}
    config_specs = {
        'nb_estimators': IntParam(default_value=100, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        rfr = RandomForestRegressor(n_estimators=params["nb_estimators"])
        rfr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = RandomForestRegressorResult(result = rfr)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("RandomForestRegressorPredictor")
class RandomForestRegressorPredictor(Task):
    """
    Predictor of a random forest regressor. Predict regression target of a dataset with a trained random forest regressor.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': RandomForestRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        rfr = learned_model.result
        y = rfr.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}