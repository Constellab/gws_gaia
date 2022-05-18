# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, IntParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from sklearn.ensemble import GradientBoostingClassifier

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# GradientBoostingClassifierResult
#
# *****************************************************************************


@resource_decorator("GradientBoostingClassifierResult", hide=True)
class GradientBoostingClassifierResult(BaseResource):
    pass

# *****************************************************************************
#
# GradientBoostingClassifierTrainer
#
# *****************************************************************************


@task_decorator("GradientBoostingClassifierTrainer", human_name="Gradient-Boosting classifier trainer",
                short_description="Train an gradient boosting classifier model")
class GradientBoostingClassifierTrainer(Task):
    """
    Trainer of a gradient boosting classifier. Fit a gradient boosting classifier with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(GradientBoostingClassifierResult, human_name="result", short_description="The output result")}
    config_specs = {
        'nb_estimators': IntParam(default_value=100, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        gbc = GradientBoostingClassifier(n_estimators=params["nb_estimators"])
        gbc.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = GradientBoostingClassifierResult(training_set=dataset, result=gbc)
        return {'result': result}

# *****************************************************************************
#
# GradientBoostingClassifierPredictor
#
# *****************************************************************************


@task_decorator("GradientBoostingClassifierPredictor", human_name="Gradient-Boosting classifier predictor",
                short_description="Predict dataset labels using a trained gradient-boosting classifier model")
class GradientBoostingClassifierPredictor(Task):
    """
    Predictor of a gradient boosting classifier. Predict classes for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(GradientBoostingClassifierResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gbc = learned_model.get_result()
        y = gbc.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
