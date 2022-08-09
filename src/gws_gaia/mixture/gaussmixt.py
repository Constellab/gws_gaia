# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from pandas import DataFrame
from sklearn.mixture import GaussianMixture

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# GaussianMixtureResult
#
# *****************************************************************************


@resource_decorator("GaussianMixtureResult", hide=True)
class GaussianMixtureResult(BaseResourceSet):
    pass

# *****************************************************************************
#
# GaussianMixtureTrainer
#
# *****************************************************************************


@task_decorator("GaussianMixtureTrainer", human_name="Gaussian mixture trainer",
                short_description="Train a Gaussian mixture model")
class GaussianMixtureTrainer(Task):
    """
    Trainer of a Gaussian mixture model. Estimate model parameters with a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit for more details
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(GaussianMixtureResult, human_name="result", short_description="The output result")}
    config_specs = {
        'nb_components': IntParam(default_value=1, min_value=0),
        'covariance_type': StrParam(default_value='full')
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        gmixt = GaussianMixture(n_components=params["nb_components"], covariance_type=params["covariance_type"])
        gmixt.fit(dataset.get_features().values)
        result = GaussianMixtureResult(training_set=dataset, result=gmixt)
        return {'result': result}

# *****************************************************************************
#
# GaussianMixturePredictor
#
# *****************************************************************************


@task_decorator("GaussianMixturePredictor", human_name="Gaussian mixture predictor",
                short_description="Predict the class labels for a dataset using a Gaussian mixture model")
class GaussianMixturePredictor(Task):
    """
    Predictor of a Gaussian mixture model. Predict the labels for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit for more details
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(GaussianMixtureResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gmixt = learned_model.get_result()
        y = gmixt.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
