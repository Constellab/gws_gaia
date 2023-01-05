# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (FloatParam, InputSpec, OutputSpec, Table,
                      resource_decorator, task_decorator)
from sklearn.naive_bayes import MultinomialNB

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedClassResult,
                             BaseSupervisedPredictor, BaseSupervisedTrainer)

# *****************************************************************************
#
# MultinomialNaiveBayesClassifierResult
#
# *****************************************************************************


@resource_decorator("MultinomialNaiveBayesClassifierResult", hide=True)
class MultinomialNaiveBayesClassifierResult(BaseSupervisedClassResult):
    pass

# *****************************************************************************
#
# MultinomialNaiveBayesClassifierTrainer
#
# *****************************************************************************


@task_decorator("MultinomialNaiveBayesClassifierTrainer", human_name="MNB trainer",
                short_description="Predict the class labels using a Multinomial Naive Bayes (MNB) classifier")
class MultinomialNaiveBayesClassifierTrainer(BaseSupervisedTrainer):
    """
    Trainer of a naive Bayes classifier for a multinomial model. Fit a naive Bayes classifier according to a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(MultinomialNaiveBayesClassifierResult,
                                         human_name="result", short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'alpha': FloatParam(default_value=1)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return MultinomialNB(alpha=params["alpha"])

    @classmethod
    def create_result_class(cls) -> Type[MultinomialNaiveBayesClassifierResult]:
        return MultinomialNaiveBayesClassifierResult

# *****************************************************************************
#
# MultinomialNaiveBayesClassifierPredictor
#
# *****************************************************************************


@task_decorator("MultinomialNaiveBayesClassifierPredictor", human_name="MNB classifier predictor",
                short_description="Predict the class labels using Multinomial Naive Bayes (MNB) classifier")
class MultinomialNaiveBayesClassifierPredictor(BaseSupervisedPredictor):
    """
    Predictor of a naïve Bayes classifier for a multinomial model. Predict class labels for a table using a trained naïve Bayes classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        MultinomialNaiveBayesClassifierResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
