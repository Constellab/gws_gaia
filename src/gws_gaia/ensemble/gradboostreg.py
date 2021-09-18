# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================


@resource_decorator("GradientBoostingRegressorResult", hide=True)
class GradientBoostingRegressorResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("GradientBoostingRegressorTrainer")
class GradientBoostingRegressorTrainer(Task):
    """
    Trainer of a gradient boosting regressor. Fit a gradient boosting regressor with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : GradientBoostingRegressorResult}
    config_specs = {
        'nb_estimators':IntParam(default_value=100, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        gbr = GradientBoostingRegressor(n_estimators=params["nb_estimators"])
        gbr.fit(dataset.features.values, ravel(dataset.targets.values))
        result = GradientBoostingRegressorResult.from_result(result=gbr)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("GradientBoostingRegressorTester")
class GradientBoostingRegressorTester(Task):
    """
    Tester of a trained gradient boosting regressor. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained gradient boosting regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GradientBoostingRegressorResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gbr = learned_model.binary_store['result']
        y = gbr.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = GenericResult.from_result(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("GradientBoostingRegressorPredictor")
class GradientBoostingRegressorPredictor(Task):
    """
    Predictor of a gradient boosting regressor. Predict regression target for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GradientBoostingRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gbr = learned_model.binary_store['result']
        y = gbr.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}