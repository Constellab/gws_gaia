# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from sklearn.mixture import GaussianMixture
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from gws_core import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("GaussianMixtureResult", hide=True)
class GaussianMixtureResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("GaussianMixtureTrainer")
class GaussianMixtureTrainer(Task):
    """
    Trainer of a Gaussian mixture model. Estimate model parameters with a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : GaussianMixtureResult}
    config_specs = {
        'nb_components': IntParam(default_value=1, min_value=0),
        'covariance_type': StrParam(default_value='full')
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        gmixt = GaussianMixture(n_components=params["nb_components"],covariance_type=params["covariance_type"])
        gmixt.fit(dataset.get_features().values)
        result = GaussianMixtureResult(result = gmixt)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("GaussianMixturePredictor")
class GaussianMixturePredictor(Task):
    """
    Predictor of a Gaussian mixture model. Predict the labels for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GaussianMixtureResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gmixt = learned_model.result
        y = gmixt.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}