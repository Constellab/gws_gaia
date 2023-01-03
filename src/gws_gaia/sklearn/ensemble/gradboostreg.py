# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator)
from sklearn.ensemble import GradientBoostingRegressor

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# GradientBoostingRegressorResult
#
# *****************************************************************************


@resource_decorator("GradientBoostingRegressorResult", hide=True)
class GradientBoostingRegressorResult(BaseSupervisedResult):
    pass

# *****************************************************************************
#
# GradientBoostingRegressorTrainer
#
# *****************************************************************************


@task_decorator("GradientBoostingRegressorTrainer", human_name="Gradient-Boosting regressor trainer",
                short_description="Train a gradient boosting regressor model")
class GradientBoostingRegressorTrainer(BaseSupervisedTrainer):
    """
    Trainer of a gradient boosting regressor. Fit a gradient boosting regressor with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(GradientBoostingRegressorResult,
                                         human_name="result", short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'nb_estimators': IntParam(default_value=100, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return GradientBoostingRegressor(n_estimators=params["nb_estimators"])

    @classmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        return GradientBoostingRegressorResult


# *****************************************************************************
#
# GradientBoostingRegressorPredictor
#
# *****************************************************************************


@task_decorator("GradientBoostingRegressorPredictor", human_name="Gradient-Boosting regressor predictor",
                short_description="Predict table targets using a trained gradient-boosting regressor model")
class GradientBoostingRegressorPredictor(BaseSupervisedPredictor):
    """
    Predictor of a gradient boosting regressor. Predict regression target for a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        GradientBoostingRegressorResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
