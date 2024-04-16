

from typing import Any, Type

from gws_core import (FloatParam, InputSpec, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.linear_model import Ridge

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedRegResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# RidgeRegressionResult
#
# *****************************************************************************


@resource_decorator("RidgeRegressionResult", hide=True)
class RidgeRegressionResult(BaseSupervisedRegResult):
    """ RidgeRegressionResult """

# *****************************************************************************
#
# RidgeRegressionTrainer
#
# *****************************************************************************


@task_decorator("RidgeRegressionTrainer", human_name="Ridge regression trainer",
                short_description="Train a ridge regression model")
class RidgeRegressionTrainer(BaseSupervisedTrainer):
    """
    Trainer of a Ridge regression model. Fit a Ridge regression model with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(RidgeRegressionResult, human_name="result",
                                         short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'alpha': FloatParam(default_value=1, min_value=0),
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return Ridge(alpha=params["alpha"])

    @classmethod
    def create_result_class(cls) -> Type[RidgeRegressionResult]:
        return RidgeRegressionResult

# *****************************************************************************
#
# RidgeRegressionPredictor
#
# *****************************************************************************


@task_decorator("RidgeRegressionPredictor", human_name="Ridge regression predictor",
                short_description="Predict table targets using a trained ridge regression model")
class RidgeRegressionPredictor(BaseSupervisedPredictor):
    """
    Predictor of a Ridge regression model. Predict target values of a table with a trained Ridge regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        RidgeRegressionResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
