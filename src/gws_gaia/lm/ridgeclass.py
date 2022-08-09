# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import RidgeClassifier

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# RidgeClassifierResult
#
# *****************************************************************************


@resource_decorator("RidgeClassifierResult", hide=True)
class RidgeClassifierResult(BaseResourceSet):
    pass

# *****************************************************************************
#
# RidgeClassifierTrainer
#
# *****************************************************************************


@task_decorator("RidgeClassifierTrainer", human_name="Ridge classifier trainer",
                short_description="Train a ridge classifier")
class RidgeClassifierTrainer(Task):
    """
    Trainer of a Ridge regression classifier. Fit a Ridge classifier model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(RidgeClassifierResult, human_name="result", short_description="The output result")}
    config_specs = {
        'alpha': FloatParam(default_value=1, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        ric = RidgeClassifier(alpha=params["alpha"])
        ric.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = RidgeClassifierResult(training_set=dataset, result=ric)
        return {'result': result}

# *****************************************************************************
#
# RidgeClassifierPredictor
#
# *****************************************************************************


@task_decorator("RidgeClassifierPredictor", human_name="Ridge classifier predictor",
                short_description="Predict class labels using a trained Ridge classifier")
class RidgeClassifierPredictor(Task):
    """
    Predictor of a Ridge regression classifier. Predict class labels for samples in a datatset with a trained Ridge classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(RidgeClassifierResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        ric = learned_model.get_result()
        y = ric.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
