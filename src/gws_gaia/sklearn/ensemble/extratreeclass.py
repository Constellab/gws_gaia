# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator)
from sklearn.ensemble import ExtraTreesClassifier

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# ExtraTreesClassifierResult
#
# *****************************************************************************


@resource_decorator("ExtraTreesClassifierResult", hide=True)
class ExtraTreesClassifierResult(BaseSupervisedResult):
    pass

# *****************************************************************************
#
# ExtraTreesClassifierTrainer
#
# *****************************************************************************


@task_decorator("ExtraTreesClassifierTrainer", human_name="Extra-Trees classifier trainer",
                short_description="Train an extra-trees classifier model")
class ExtraTreesClassifierTrainer(BaseSupervisedTrainer):
    """
    Trainer of an extra-trees classifier. Build a forest of trees from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier.fit for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(ExtraTreesClassifierResult,
                                         human_name="result", short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'nb_estimators': IntParam(default_value=100, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Type[Any]:
        return ExtraTreesClassifier(n_estimators=params["nb_estimators"])

    @classmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        return ExtraTreesClassifierResult

# *****************************************************************************
#
# ExtraTreesClassifierPredictor
#
# *****************************************************************************


@task_decorator("ExtraTreesClassifierPredictor", human_name="Extra-Trees classifier predictor",
                short_description="Predict table labels using a trained extra-trees classifier model")
class ExtraTreesClassifierPredictor(BaseSupervisedPredictor):
    """
    Predictor of an extra-trees classifier. Predict class for a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier.fit for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        ExtraTreesClassifierResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
