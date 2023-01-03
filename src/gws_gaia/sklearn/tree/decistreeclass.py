# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (ConfigParams, FloatParam, InputSpec, IntParam,
                      OutputSpec, Resource, StrParam, Table, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# DecisionTreeClassifierResult
#
# *****************************************************************************


@resource_decorator("DecisionTreeClassifierResult", hide=True)
class DecisionTreeClassifierResult(BaseSupervisedResult):
    pass

# *****************************************************************************
#
# DecisionTreeClassifierTrainer
#
# *****************************************************************************


@task_decorator("DecisionTreeClassifierTrainer", human_name="Decision tree classifier trainer",
                short_description="Train a decision tree classifier model")
class DecisionTreeClassifierTrainer(BaseSupervisedTrainer):
    """ Trainer of the decision tree classifier. Build a decision tree classifier from the training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(DecisionTreeClassifierResult,
                                         human_name="result", short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'max_depth': IntParam(default_value=None, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return DecisionTreeClassifier(max_depth=params["max_depth"])

    @classmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        return DecisionTreeClassifierResult

# *****************************************************************************
#
# DecisionTreeClassifierPredictor
#
# *****************************************************************************


@task_decorator("DecisionTreeClassifierPredictor", human_name="Decision tree classifier predictor",
                short_description="Predict class labels for a table using a decision tree classifier model")
class DecisionTreeClassifierPredictor(BaseSupervisedPredictor):
    """ Predictor of a trained decision tree classifier.
    Predict class for a table. The predicted class for each sample in the table is returned.

    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        DecisionTreeClassifierResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
