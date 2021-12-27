# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import AdaBoostRegressor

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from gws_core import Dataset
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# AdaBoostRegressorResult
#
# *****************************************************************************

@resource_decorator("AdaBoostRegressorResult", hide=True)
class AdaBoostRegressorResult(BaseResource):
    pass

# *****************************************************************************
#
# AdaBoostRegressorTrainer
#
# *****************************************************************************

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
        abr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = AdaBoostRegressorResult(result = abr)
        return {'result': result}

# *****************************************************************************
#
# AdaBoostRegressorPredictor
#
# *****************************************************************************

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
        abr = learned_model.result
        y = abr.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y, 
            row_names=dataset.row_names, 
            column_names=dataset.target_names, 
            target_names=dataset.target_names
        )
        return {'result': result_dataset}