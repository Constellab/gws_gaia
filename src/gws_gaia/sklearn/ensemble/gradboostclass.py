

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.ensemble import GradientBoostingClassifier

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedClassResult,
                             BaseSupervisedPredictor, BaseSupervisedTrainer)

# *****************************************************************************
#
# GradientBoostingClassifierResult
#
# *****************************************************************************


@resource_decorator("GradientBoostingClassifierResult", hide=True)
class GradientBoostingClassifierResult(BaseSupervisedClassResult):
    pass

# *****************************************************************************
#
# GradientBoostingClassifierTrainer
#
# *****************************************************************************


@task_decorator("GradientBoostingClassifierTrainer", human_name="Gradient-Boosting classifier trainer",
                short_description="Train an gradient boosting classifier model")
class GradientBoostingClassifierTrainer(BaseSupervisedTrainer):
    """
    Trainer of a gradient boosting classifier. Fit a gradient boosting classifier with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(GradientBoostingClassifierResult,
                                         human_name="result", short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'nb_estimators': IntParam(default_value=100, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return GradientBoostingClassifier(n_estimators=params["nb_estimators"])

    @classmethod
    def create_result_class(cls) -> Type[GradientBoostingClassifierResult]:
        return GradientBoostingClassifierResult

# *****************************************************************************
#
# GradientBoostingClassifierPredictor
#
# *****************************************************************************


@task_decorator("GradientBoostingClassifierPredictor", human_name="Gradient-Boosting classifier predictor",
                short_description="Predict table labels using a trained gradient-boosting classifier model")
class GradientBoostingClassifierPredictor(BaseSupervisedPredictor):
    """
    Predictor of a gradient boosting classifier. Predict classes for a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        GradientBoostingClassifierResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
