# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.decomposition import FastICA

from ..base.base_unsup import BaseUnsupervisedResult, BaseUnsupervisedTrainer

# *****************************************************************************
#
# ICAResult
#
# *****************************************************************************


@resource_decorator("ICAResult", hide=True)
class ICAResult(BaseUnsupervisedResult):
    pass

# *****************************************************************************
#
# ICATrainer
#
# *****************************************************************************


@task_decorator("ICATrainer", human_name="ICA trainer",
                short_description="Train an Independant Component Analysis (ICA) model")
class ICATrainer(BaseUnsupervisedTrainer):
    """
    Trainer of an Independant Component Analysis (ICA) model. Fit a model of ICA to a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.ICA.html#sklearn.decomposition.ICA.fit for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(ICAResult, human_name="result", short_description="The output result")})
    config_specs = {
        'nb_components': IntParam(default_value=2, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Type[Any]:
        return FastICA(n_components=params["nb_components"])

    @classmethod
    def create_result_class(cls) -> Type[ICAResult]:
        return ICAResult
