# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator)
from sklearn.ensemble import ExtraTreesRegressor

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# ExtraTreesRegressorResult
#
# *****************************************************************************


@resource_decorator("ExtraTreesRegressorResult", hide=True)
class ExtraTreesRegressorResult(BaseSupervisedResult):
    pass

# *****************************************************************************
#
# ExtraTreesRegressorTrainer
#
# *****************************************************************************


@task_decorator("ExtraTreesRegressorTrainer", human_name="Extra-Trees regressor trainer",
                short_description="Train an extra-trees regressor model")
class ExtraTreesRegressorTrainer(BaseSupervisedTrainer):
    """
    Trainer of an extra-trees regressor. Build a forest of trees from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(ExtraTreesRegressorResult, human_name="result",
                                         short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'nb_estimators': IntParam(default_value=100, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return ExtraTreesRegressor(n_estimators=params["nb_estimators"])

    @classmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        return ExtraTreesRegressorResult

# *****************************************************************************
#
# ExtraTreesRegressorPredictor
#
# *****************************************************************************


@task_decorator("ExtraTreesRegressorPredictor", human_name="Extra-Trees regressor predictor",
                short_description="Predict table targets using a trained extra-trees regressor model")
class ExtraTreesRegressorPredictor(BaseSupervisedPredictor):
    """
    Predictor of an extra-trees regressor. Predict regression target of a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        ExtraTreesRegressorResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
