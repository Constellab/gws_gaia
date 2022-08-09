# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, IntParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from sklearn.ensemble import AdaBoostRegressor

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# AdaBoostRegressorResult
#
# *****************************************************************************


@resource_decorator("AdaBoostRegressorResult", hide=True)
class AdaBoostRegressorResult(BaseResourceSet):
    pass

# *****************************************************************************
#
# AdaBoostRegressorTrainer
#
# *****************************************************************************


@task_decorator("AdaBoostRegressorTrainer", human_name="AdaBoost regression trainer",
                short_description="Train an AdaBoost regression model")
class AdaBoostRegressorTrainer(Task):
    """
    Trainer of an Adaboost regressor. This process build a boosted regressor from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html for more details
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(AdaBoostRegressorResult, human_name="result", short_description="The output result")}
    config_specs = {
        'nb_estimators': IntParam(default_value=50, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        abr = AdaBoostRegressor(n_estimators=params["nb_estimators"])
        abr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = AdaBoostRegressorResult(training_set=dataset, result=abr)
        return {'result': result}

# *****************************************************************************
#
# AdaBoostRegressorPredictor
#
# *****************************************************************************


@task_decorator("AdaBoostRegressorPredictor", human_name="AdaBoost regression predictor",
                short_description="Predict dataset targets using a trained AdaBoost regression model")
class AdaBoostRegressorPredictor(Task):
    """
    Predictor of a trained Adaboost regressor. The predicted regression value of an input sample is computed as the weighted median
    prediction of the classifiers in the ensemble.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html for more details
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(AdaBoostRegressorResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        abr = learned_model.get_result()
        y = abr.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
