# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator)
from sklearn.ensemble import RandomForestClassifier

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# RandomForestClassifierResult
#
# *****************************************************************************


@resource_decorator("RandomForestClassifierResult", hide=True)
class RandomForestClassifierResult(BaseSupervisedResult):
    pass

# *****************************************************************************
#
# RandomForestClassifierTrainer
#
# *****************************************************************************


@task_decorator("RandomForestClassifierTrainer", human_name="Random-Forest classifier trainer",
                short_description="Train a random forest classifier model")
class RandomForestClassifierTrainer(BaseSupervisedTrainer):
    """
    Trainer of a random forest classifier. Build a forest of trees from a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(RandomForestClassifierResult,
                                         human_name="result", short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'nb_estimators': IntParam(default_value=100, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return RandomForestClassifier(n_estimators=params["nb_estimators"])

    @classmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        return RandomForestClassifierResult


# *****************************************************************************
#
# RandomForestClassifierPredictor
#
# *****************************************************************************


@task_decorator("RandomForestClassifierPredictor", human_name="Random-Forest classifier predictor",
                short_description="Predict table labels using a trained Random forest classifier model")
class RandomForestClassifierPredictor(BaseSupervisedPredictor):
    """
    Predictor of a random forest classifier. Predict class labels of a table with a trained random forest classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        RandomForestClassifierResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
