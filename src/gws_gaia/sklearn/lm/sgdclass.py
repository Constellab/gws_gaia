# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (FloatParam, InputSpec, IntParam, OutputSpec, StrParam,
                      Table, resource_decorator, task_decorator)
from sklearn.linear_model import SGDClassifier

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedClassResult,
                             BaseSupervisedPredictor, BaseSupervisedTrainer)

# *****************************************************************************
#
# SGDClassifierResult
#
# *****************************************************************************


@resource_decorator("SGDClassifierResult")
class SGDClassifierResult(BaseSupervisedClassResult):
    """ RidgeClassifierResult """

# *****************************************************************************
#
# SGDClassifierTrainer
#
# *****************************************************************************


@task_decorator("SGDClassifierTrainer", human_name="SGD classifier trainer",
                short_description="Train a stochastic gradient descent (SGD) linear classifier")
class SGDClassifierTrainer(BaseSupervisedTrainer):
    """
    Trainer of a linear classifier with stochastic gradient descent (SGD). Fit a SGD linear classifier with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(SGDClassifierResult, human_name="result",
                                         short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'loss':
        StrParam(
            default_value='hinge',
            allowed_values=['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron',
                            'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
        'alpha': FloatParam(default_value=0.0001, min_value=0),
        'max_iter': IntParam(default_value=1000, min_value=0), }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return SGDClassifier(max_iter=params["max_iter"], alpha=params["alpha"], loss=params["loss"])

    @classmethod
    def create_result_class(cls) -> Type[SGDClassifierResult]:
        return SGDClassifierResult

# *****************************************************************************
#
# SGDClassifierPredictor
#
# *****************************************************************************


@task_decorator("SGDClassifierPredictor", human_name="SGD classifier predictor",
                short_description="Predict class labels using a trained SGD classifier")
class SGDClassifierPredictor(BaseSupervisedPredictor):
    """
    Predictor of a linear classifier with stochastic gradient descent (SGD). Predict class labels of a table with a trained SGD linear classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        SGDClassifierResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
