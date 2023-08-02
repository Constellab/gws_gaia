# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.gaussian_process import GaussianProcessClassifier

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedClassResult,
                             BaseSupervisedPredictor, BaseSupervisedTrainer)

# *****************************************************************************
#
# GaussianProcessClassifierResult
#
# *****************************************************************************


@resource_decorator("GaussianProcessClassifierResult", hide=True)
class GaussianProcessClassifierResult(BaseSupervisedClassResult):
    """ GaussianProcessClassifierResult """

# *****************************************************************************
#
# GaussianProcessClassifierTrainer
#
# *****************************************************************************


@task_decorator("GaussianProcessClassifierTrainer", human_name="Gaussian process classifier trainer",
                short_description="Train a gaussian process classifier model")
class GaussianProcessClassifierTrainer(BaseSupervisedTrainer):
    """
    Trainer of a Gaussian process classifier. Fit a Gaussian process classification model with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier.fit for more details
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(GaussianProcessClassifierResult,
                                         human_name="result", short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'random_state': IntParam(default_value=None, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return GaussianProcessClassifier(random_state=params["random_state"])

    @classmethod
    def create_result_class(cls) -> Type[GaussianProcessClassifierResult]:
        return GaussianProcessClassifierResult

# *****************************************************************************
#
# GaussianProcessClassifierPredictor
#
# *****************************************************************************


@task_decorator("GaussianProcessClassifierPredictor", human_name="Gaussian process classifier predictor",
                short_description="Predict table labels using a trained gaussian process classifier model")
class GaussianProcessClassifierPredictor(BaseSupervisedPredictor):
    """
    Predictor of a Gaussian process classifier. Predict classes of a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier.fit for more details
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        GaussianProcessClassifierResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
