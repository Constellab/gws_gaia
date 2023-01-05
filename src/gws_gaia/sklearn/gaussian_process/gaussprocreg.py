# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (FloatParam, InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator)
from sklearn.gaussian_process import GaussianProcessRegressor

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# GaussianProcessRegressorResult
#
# *****************************************************************************


@resource_decorator("GaussianProcessRegressorResult", hide=True)
class GaussianProcessRegressorResult(BaseSupervisedResult):
    pass

# *****************************************************************************
#
# GaussianProcessRegressorTrainer
#
# *****************************************************************************


@task_decorator("GaussianProcessRegressorTrainer", human_name="Gaussian process regressor trainer",
                short_description="Train a Gaussian process regression model")
class GaussianProcessRegressorTrainer(BaseSupervisedTrainer):
    """
    Trainer of a Gaussian process regressor. Fit a Gaussian process regressor model with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(GaussianProcessRegressorResult,
                                         human_name="result", short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'alpha': FloatParam(default_value=1e-10, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return GaussianProcessRegressor(alpha=params["alpha"])

    @classmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        return GaussianProcessRegressorResult

# *****************************************************************************
#
# GaussianProcessRegressorPredictor
#
# *****************************************************************************


@task_decorator("GaussianProcessRegressorPredictor", human_name="Gaussian process regressor predictor",
                short_description="Predict table targets using a trained Gaussian process regression model")
class GaussianProcessRegressorPredictor(BaseSupervisedPredictor):
    """
    Predictor of a Gaussian process regressor. Predict regression targets of a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        GaussianProcessRegressorResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}