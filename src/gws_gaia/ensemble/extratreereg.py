# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import ExtraTreesRegressor

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from gws_core import Dataset
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# ExtraTreesRegressorResult
#
# *****************************************************************************

@resource_decorator("ExtraTreesRegressorResult", hide=True)
class ExtraTreesRegressorResult(BaseResource):
    pass

# *****************************************************************************
#
# ExtraTreesRegressorTrainer
#
# *****************************************************************************

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
        etr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = ExtraTreesRegressorResult(result = etr)
        return {'result': result}

# *****************************************************************************
#
# ExtraTreesRegressorPredictor
#
# *****************************************************************************

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
        etr = learned_model.result
        y = etr.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y, 
            row_names=dataset.row_names, 
            column_names=dataset.target_names, 
            target_names=dataset.target_names
        )
        return {'result': result_dataset}