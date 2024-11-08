

from typing import Any, Type

from gws_core import (FloatParam, InputSpec, IntParam, OutputSpec, StrParam,
                      Table, resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.linear_model import SGDRegressor

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedRegResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# SGDRegressorResult
#
# *****************************************************************************


@resource_decorator("SGDRegressorResult", hide=True)
class SGDRegressorResult(BaseSupervisedRegResult):
    """ SGDRegressorResult """

# *****************************************************************************
#
# SGDRegressorTrainer
#
# *****************************************************************************


@task_decorator("SGDRegressorTrainer", human_name="SGD regressor trainer",
                short_description="Train a stochastic gradient descent (SGD) linear regressor")
class SGDRegressorTrainer(BaseSupervisedTrainer):
    """
    Trainer of a linear regressor with stochastic gradient descent (SGD). Fit a SGD linear regressor with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(SGDRegressorResult, human_name="result",
                                         short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'loss':
        StrParam(
            default_value='squared_error',
            allowed_values=['squared_error', 'epsilon_insensitive', 'huber', 'squared_epsilon_insensitive']),
        'alpha': FloatParam(default_value=0.0001, min_value=0),
        'max_iter': IntParam(default_value=1000, min_value=0)}

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return SGDRegressor(max_iter=params["max_iter"], alpha=params["alpha"], loss=params["loss"])

    @classmethod
    def create_result_class(cls) -> Type[SGDRegressorResult]:
        return SGDRegressorResult

# *****************************************************************************
#
# SGDRegressorPredictor
#
# *****************************************************************************


@task_decorator("SGDRegressorPredictor", human_name="SGD regressor predictor",
                short_description="Predict targets using a trained SGD linear regressor")
class SGDRegressorPredictor(BaseSupervisedPredictor):
    """
    Predictor of a linear regressor with stochastic gradient descent (SGD). Predict target values of a table with a trained SGD linear regressor.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        SGDRegressorResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
