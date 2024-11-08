

from typing import Any, Type

from gws_core import (FloatParam, InputSpec, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedClassResult,
                             BaseSupervisedPredictor, BaseSupervisedTrainer)

# *****************************************************************************
#
# QDAResult
#
# *****************************************************************************


@resource_decorator("QDAResult", human_name="QDA result", hide=True)
class QDAResult(BaseSupervisedClassResult):
    pass

# *****************************************************************************
#
# QDATrainer
#
# *****************************************************************************


@task_decorator("QDATrainer", human_name="QDA trainer",
                short_description="Train a Quadratic Discriminant Analysis (QDA) model")
class QDATrainer(BaseSupervisedTrainer):
    """
    Trainer of quadratic discriminant analysis model. Fit a quadratic discriminant analysis model with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(QDAResult, human_name="result", short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'reg_param': FloatParam(default_value=0),
    }

    @classmethod
    def create_classifier_class(cls, params) -> Type[Any]:
        return QuadraticDiscriminantAnalysis(reg_param=params["reg_param"])

    @classmethod
    def create_result_class(cls) -> Type[QDAResult]:
        return QDAResult


# *****************************************************************************
#
# QDAPredictor
#
# *****************************************************************************


@task_decorator("QDAPredictor", human_name="QDA predictor",
                short_description="Predict class labels using a Quadratic Discriminant Analysis (QDA) model")
class QDAPredictor(BaseSupervisedPredictor):
    """
    Predictor of quadratic discriminant analysis model. Predic class labels of a table with a trained quadratic discriminant analysis model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html for more details.
    """
    input_specs = InputSpecs({
        'table': InputSpec(Table, human_name="Table", short_description="The input table"),
        'learned_model': InputSpec(QDAResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
