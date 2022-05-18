# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, IntParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from sklearn.gaussian_process import GaussianProcessClassifier

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# GaussianProcessClassifierResult
#
# *****************************************************************************


@resource_decorator("GaussianProcessClassifierResult", hide=True)
class GaussianProcessClassifierResult(BaseResource):
    pass

# *****************************************************************************
#
# GaussianProcessClassifierTrainer
#
# *****************************************************************************


@task_decorator("GaussianProcessClassifierTrainer", human_name="Gaussian process classifier trainer",
                short_description="Train a gaussian process classifier model")
class GaussianProcessClassifierTrainer(Task):
    """
    Trainer of a Gaussian process classifier. Fit a Gaussian process classification model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier.fit for more details
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(GaussianProcessClassifierResult, human_name="result", short_description="The output result")}
    config_specs = {
        'random_state': IntParam(default_value=None, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        gpc = GaussianProcessClassifier(random_state=params["random_state"])
        gpc.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = GaussianProcessClassifierResult(training_set=dataset, result=gpc)
        return {'result': result}

# *****************************************************************************
#
# GaussianProcessClassifierPredictor
#
# *****************************************************************************


@task_decorator("GaussianProcessClassifierPredictor", human_name="Gaussian process classifier predictor",
                short_description="Predict dataset labels using a trained gaussian process classifier model")
class GaussianProcessClassifierPredictor(Task):
    """
    Predictor of a Gaussian process classifier. Predict classes of a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier.fit for more details
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(GaussianProcessClassifierResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gpc = learned_model.get_result()
        y = gpc.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
