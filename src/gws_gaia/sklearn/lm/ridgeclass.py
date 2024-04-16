

from typing import Any, Type

from gws_core import (FloatParam, InputSpec, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.linear_model import RidgeClassifier

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedClassResult,
                             BaseSupervisedPredictor, BaseSupervisedTrainer)

# *****************************************************************************
#
# RidgeClassifierResult
#
# *****************************************************************************


@resource_decorator("RidgeClassifierResult", hide=True)
class RidgeClassifierResult(BaseSupervisedClassResult):
    """ RidgeClassifierResult """

# *****************************************************************************
#
# RidgeClassifierTrainer
#
# *****************************************************************************


@task_decorator("RidgeClassifierTrainer", human_name="Ridge classifier trainer",
                short_description="Train a ridge classifier")
class RidgeClassifierTrainer(BaseSupervisedTrainer):
    """
    Trainer of a Ridge regression classifier. Fit a Ridge classifier model with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(RidgeClassifierResult, human_name="result",
                                         short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'alpha': FloatParam(default_value=1, min_value=0),
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return RidgeClassifier(alpha=params["alpha"])

    @classmethod
    def create_result_class(cls) -> Type[RidgeClassifierResult]:
        return RidgeClassifierResult

# *****************************************************************************
#
# RidgeClassifierPredictor
#
# *****************************************************************************


@task_decorator("RidgeClassifierPredictor", human_name="Ridge classifier predictor",
                short_description="Predict class labels using a trained Ridge classifier")
class RidgeClassifierPredictor(BaseSupervisedPredictor):
    """
    Predictor of a Ridge regression classifier. Predict class labels for samples in a datatset with a trained Ridge classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        RidgeClassifierResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
