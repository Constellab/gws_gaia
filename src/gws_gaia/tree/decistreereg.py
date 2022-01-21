# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from numpy import ravel
from pandas import DataFrame
from sklearn.tree import DecisionTreeRegressor

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from gws_core import Dataset
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# DecisionTreeRegressorResult
#
# *****************************************************************************

@resource_decorator("DecisionTreeRegressorResult")
class DecisionTreeRegressorResult(BaseResource):
    pass

# *****************************************************************************
#
# DecisionTreeRegressorTrainer
#
# *****************************************************************************

@task_decorator("DecisionTreeRegressorTrainer", human_name="Decision tree regressor trainer",
                short_description="Train a decision tree regressor model")
class DecisionTreeRegressorTrainer(Task):
    """ Trainer of a decision tree regressor. Build a decision tree regressor from a training dataset

    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : DecisionTreeRegressorResult}
    config_specs = {
        'max_depth':IntParam(default_value=None, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        dtr = DecisionTreeRegressor(max_depth=params["max_depth"])
        dtr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = DecisionTreeRegressorResult(result = dtr)
        return {'result': result}

# *****************************************************************************
#
# DecisionTreeRegressorPredictor
#
# *****************************************************************************

@task_decorator("DecisionTreeRegressorPredictor", human_name="Decision tree regressor predictor",
                short_description="Predict targets for a dataset using a decision tree regressor model")
class DecisionTreeRegressorPredictor(Task):
    """ Predictor of a trained decision tree regressor.
    Predict regression value for the dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': DecisionTreeRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        dtr = learned_model.result
        y = dtr.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}