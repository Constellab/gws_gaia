# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import AdaBoostRegressor

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.dataset import Dataset
from ..data.core import Tuple
from ..base.base_resource import BaseResource

@resource_decorator("AdaBoostRegressorResult", hide=True)
class AdaBoostRegressorResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("AdaBoostRegressorTrainer")
class AdaBoostRegressorTrainer(Task):
    """
    Trainer of an Adaboost regressor. This process build a boosted regressor from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : AdaBoostRegressorResult}
    config_specs = {
        'nb_estimators':IntParam(default_value=50, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        abr = AdaBoostRegressor(n_estimators=params["nb_estimators"])
        abr.fit(dataset.features.values, ravel(dataset.targets.values))
        result = AdaBoostRegressorResult.from_result(result=abr)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("AdaBoostRegressorTester")
class AdaBoostRegressorTester(Task):
    """
    Tester of a trained Adaboost regressor. Return the coefficient of determination R^2 of the prediction on a given test data for a trained Adaboost regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': AdaBoostRegressorResult}
    output_specs = {'result' : Tuple}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        abr = learned_model.binary_store['result']
        y = abr.score(dataset.features.values,ravel(dataset.targets.values))
        z = tuple([y])
        result_dataset = Tuple(tup = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("AdaBoostRegressorPredictor")
class AdaBoostRegressorPredictor(Task):
    """
    Predictor of a trained Adaboost regressor. The predicted regression value of an input sample is computed as the weighted median 
    prediction of the classifiers in the ensemble.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': AdaBoostRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        abr = learned_model.binary_store['result']
        y = abr.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}