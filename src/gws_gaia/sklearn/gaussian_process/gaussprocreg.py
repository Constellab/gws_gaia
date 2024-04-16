

from typing import Any, Type

from gws_core import (FloatParam, InputSpec, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.gaussian_process import GaussianProcessRegressor

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedRegResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# GaussianProcessRegressorResult
#
# *****************************************************************************


@resource_decorator("GaussianProcessRegressorResult", hide=True)
class GaussianProcessRegressorResult(BaseSupervisedRegResult):
    """ GaussianProcessRegressorResult """

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
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(GaussianProcessRegressorResult,
                                         human_name="result", short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'alpha': FloatParam(default_value=1e-10, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return GaussianProcessRegressor(alpha=params["alpha"])

    @classmethod
    def create_result_class(cls) -> Type[GaussianProcessRegressorResult]:
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
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        GaussianProcessRegressorResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
