# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.ensemble import AdaBoostClassifier

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedClassResult,
                             BaseSupervisedPredictor, BaseSupervisedTrainer)

# *****************************************************************************
#
# AdaBoostClassifierResult
#
# *****************************************************************************


@resource_decorator("AdaBoostClassifierResult", hide=True)
class AdaBoostClassifierResult(BaseSupervisedClassResult):
    """AdaBoostClassifierResult"""

# *****************************************************************************
#
# AdaBoostClassifierTrainer
#
# *****************************************************************************


@task_decorator("AdaBoostClassifierTrainer", human_name="AdaBoost classifier trainer",
                short_description="Train an AdaBoost classifier model")
class AdaBoostClassifierTrainer(BaseSupervisedTrainer):
    """
    Trainer of an AdaBoost classifier. This process builds a boosted classifier from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html for more details
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(AdaBoostClassifierResult, human_name="result",
                                         short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'nb_estimators': IntParam(default_value=50, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Type[Any]:
        return AdaBoostClassifier(n_estimators=params["nb_estimators"])

    @classmethod
    def create_result_class(cls) -> Type[AdaBoostClassifierResult]:
        return AdaBoostClassifierResult


# *****************************************************************************
#
# AdaBoostClassifierPredictor
#
# *****************************************************************************


@task_decorator("AdaBoostClassifierPredictor", human_name="AdaBoost classifier predictor",
                short_description="Predict table labels using a trained AdaBoost classifier model")
class AdaBoostClassifierPredictor(BaseSupervisedPredictor):
    """
    Predictor of a trained AdaBoost classifier. This process predicts classes for a table.
    The predicted class of an input sample is computed as the weighted mean prediction of the classifiers in the ensemble.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html for more details
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        AdaBoostClassifierResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
