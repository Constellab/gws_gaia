# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, InputSpecs, IntParam, OutputSpec, OutputSpecs,
                      Table, resource_decorator, task_decorator)
from sklearn.tree import DecisionTreeRegressor

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedRegResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# DecisionTreeRegressorResult
#
# *****************************************************************************


@resource_decorator("DecisionTreeRegressorResult")
class DecisionTreeRegressorResult(BaseSupervisedRegResult):
    pass

# *****************************************************************************
#
# DecisionTreeRegressorTrainer
#
# *****************************************************************************


@task_decorator("DecisionTreeRegressorTrainer", human_name="Decision tree regressor trainer",
                short_description="Train a decision tree regressor model")
class DecisionTreeRegressorTrainer(BaseSupervisedTrainer):
    """ Trainer of a decision tree regressor. Build a decision tree regressor from a training table

    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html for more details
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(DecisionTreeRegressorResult,
                                                     human_name="result", short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'max_depth': IntParam(default_value=None, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return DecisionTreeRegressor(max_depth=params["max_depth"])

    @classmethod
    def create_result_class(cls) -> Type[DecisionTreeRegressorResult]:
        return DecisionTreeRegressorResult

# *****************************************************************************
#
# DecisionTreeRegressorPredictor
#
# *****************************************************************************


@task_decorator("DecisionTreeRegressorPredictor", human_name="Decision tree regressor predictor",
                short_description="Predict targets for a table using a decision tree regressor model")
class DecisionTreeRegressorPredictor(BaseSupervisedPredictor):
    """ Predictor of a trained decision tree regressor.
    Predict regression value for the table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html for more details
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        DecisionTreeRegressorResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(
        Table, human_name="result", short_description="The output result")})
    config_specs = {}
