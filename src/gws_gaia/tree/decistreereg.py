# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from pandas import DataFrame
from sklearn.tree import DecisionTreeRegressor

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# DecisionTreeRegressorResult
#
# *****************************************************************************


@resource_decorator("DecisionTreeRegressorResult")
class DecisionTreeRegressorResult(BaseResourceSet):
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
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(DecisionTreeRegressorResult, human_name="result", short_description="The output result")}
    config_specs = {
        'max_depth': IntParam(default_value=None, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        dtr = DecisionTreeRegressor(max_depth=params["max_depth"])
        dtr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = DecisionTreeRegressorResult(training_set=dataset, result=dtr)
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
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(DecisionTreeRegressorResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        dtr = learned_model.get_result()
        y = dtr.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
