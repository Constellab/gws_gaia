# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from gws_core import (ConfigParams, Dataset, IntParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from sklearn.ensemble import GradientBoostingRegressor

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# GradientBoostingRegressorResult
#
# *****************************************************************************


@resource_decorator("GradientBoostingRegressorResult", hide=True)
class GradientBoostingRegressorResult(BaseResource):
    pass

# *****************************************************************************
#
# GradientBoostingRegressorTrainer
#
# *****************************************************************************


@task_decorator("GradientBoostingRegressorTrainer", human_name="Gradient-Boosting regressor trainer",
                short_description="Train a gradient boosting regressor model")
class GradientBoostingRegressorTrainer(Task):
    """
    Trainer of a gradient boosting regressor. Fit a gradient boosting regressor with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(GradientBoostingRegressorResult, human_name="result", short_description="The output result")}
    config_specs = {
        'nb_estimators': IntParam(default_value=100, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        gbr = GradientBoostingRegressor(n_estimators=params["nb_estimators"])
        gbr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = GradientBoostingRegressorResult(training_set=dataset, result=gbr)
        return {'result': result}

# *****************************************************************************
#
# GradientBoostingRegressorPredictor
#
# *****************************************************************************


@task_decorator("GradientBoostingRegressorPredictor", human_name="Gradient-Boosting regressor predictor",
                short_description="Predict dataset targets using a trained gradient-boosting regressor model")
class GradientBoostingRegressorPredictor(Task):
    """
    Predictor of a gradient boosting regressor. Predict regression target for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(GradientBoostingRegressorResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gbr = learned_model.get_result()
        y = gbr.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
