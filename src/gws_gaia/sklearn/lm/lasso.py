

from typing import Any, Type

from gws_core import (FloatParam, InputSpec, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.linear_model import Lasso

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedRegResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# LassoResult
#
# *****************************************************************************


@resource_decorator("LassoResult", hide=True)
class LassoResult(BaseSupervisedRegResult):
    """ LassoResult"""

# *****************************************************************************
#
# LassoTrainer
#
# *****************************************************************************


@task_decorator("LassoTrainer", human_name="Lasso trainer",
                short_description="Train a Lasso model")
class LassoTrainer(BaseSupervisedTrainer):
    """
    Trainer of a lasso model. Fit a lasso model with a training table.

    See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.Lasso.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(LassoResult, human_name="result", short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'alpha': FloatParam(default_value=1, min_value=0),
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return Lasso(alpha=params["alpha"])

    @classmethod
    def create_result_class(cls) -> Type[LassoResult]:
        return LassoResult

# *****************************************************************************
#
# LassoPredictor
#
# *****************************************************************************


@task_decorator("LassoPredictor", human_name="Lasso predictor",
                short_description="Predict table targets using a trained Lasso model")
class LassoPredictor(BaseSupervisedPredictor):
    """
    Predictor of a lasso model. Predict target values from a table with a trained lasso model.

    See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.Lasso.html for more details.
    """
    input_specs = InputSpecs({
        'table': InputSpec(Table, human_name="Table", short_description="The input table"),
        'learned_model': InputSpec(LassoResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
