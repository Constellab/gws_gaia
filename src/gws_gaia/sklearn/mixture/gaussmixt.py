# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, StrParam, Table,
                      resource_decorator, task_decorator)
from sklearn.mixture import GaussianMixture

from ..base.base_unsup import BaseUnsupervisedResult, BaseUnsupervisedTrainer

# *****************************************************************************
#
# GaussianMixtureResult
#
# *****************************************************************************


@resource_decorator("GaussianMixtureResult", hide=True)
class GaussianMixtureResult(BaseUnsupervisedResult):
    pass

# *****************************************************************************
#
# GaussianMixtureTrainer
#
# *****************************************************************************


@task_decorator("GaussianMixtureTrainer", human_name="Gaussian mixture trainer",
                short_description="Train a Gaussian mixture model")
class GaussianMixtureTrainer(BaseUnsupervisedTrainer):
    """
    Trainer of a Gaussian mixture model. Estimate model parameters with a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(GaussianMixtureResult, human_name="result",
                                         short_description="The output result")}
    config_specs = {
        'nb_components': IntParam(default_value=1, min_value=0),
        'covariance_type': StrParam(default_value='full')
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return GaussianMixture(n_components=params["nb_components"], covariance_type=params["covariance_type"])

    @classmethod
    def create_result_class(cls) -> Type[GaussianMixtureResult]:
        return GaussianMixtureResult
