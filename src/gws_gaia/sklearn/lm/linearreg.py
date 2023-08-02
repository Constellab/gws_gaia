# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, OutputSpec, Table, resource_decorator,
                      task_decorator, InputSpecs, OutputSpecs)
from sklearn.linear_model import LinearRegression

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedRegResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# LinearRegressionResult
#
# *****************************************************************************


@resource_decorator("LinearRegressionResult", hide=True)
class LinearRegressionResult(BaseSupervisedRegResult):
    """LinearRegressionResult"""

# *****************************************************************************
#
# LinearRegressionTrainer
#
# *****************************************************************************


@task_decorator("LinearRegressionTrainer", human_name="Linear regression trainer",
                short_description="Train a linear regression model")
class LinearRegressionTrainer(BaseSupervisedTrainer):
    """
    Trainer fo a linear regression model. Fit a linear regression model with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(LinearRegressionResult, human_name="result",
                                         short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return LinearRegression()

    @classmethod
    def create_result_class(cls) -> Type[LinearRegressionResult]:
        return LinearRegressionResult

# *****************************************************************************
#
# LinearRegressionPredictor
#
# *****************************************************************************


@task_decorator("LinearRegressionPredictor", human_name="Linear regression predictor",
                short_description="Predict table targets using a trained linear regression model")
class LinearRegressionPredictor(BaseSupervisedPredictor):
    """
    Predictor of a linear regression model. Predict target values of a table with a trained linear regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        LinearRegressionResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
