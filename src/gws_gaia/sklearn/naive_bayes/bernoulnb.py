# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (FloatParam, InputSpec, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.naive_bayes import BernoulliNB

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedClassResult,
                             BaseSupervisedPredictor, BaseSupervisedTrainer)

# *****************************************************************************
#
# BernoulliNaiveBayesClassifierResult
#
# *****************************************************************************


@resource_decorator("BernoulliNaiveBayesClassifierResult", hide=True)
class BernoulliNaiveBayesClassifierResult(BaseSupervisedClassResult):
    """ BernoulliNaiveBayesClassifierResult """

# *****************************************************************************
#
# BernoulliNaiveBayesClassifierTrainer
#
# *****************************************************************************


@task_decorator("BernoulliNaiveBayesClassifierTrainer", human_name="BNB classifier trainer",
                short_description="Train a Bernoulli naive Bayes (BNB) classifier model")
class BernoulliNaiveBayesClassifierTrainer(BaseSupervisedTrainer):
    """
    Trainer of a Naive Bayes classifier. Fit Naive Bayes classifier with table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html for more details
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(BernoulliNaiveBayesClassifierResult,
                                         human_name="result", short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'alpha': FloatParam(default_value=1)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return BernoulliNB(alpha=params["alpha"])

    @classmethod
    def create_result_class(cls) -> Type[BernoulliNaiveBayesClassifierResult]:
        return BernoulliNaiveBayesClassifierResult

# *****************************************************************************
#
# BernoulliNaiveBayesClassifierPredictor
#
# *****************************************************************************


@task_decorator("BernoulliNaiveBayesClassifierPredictor", human_name="BNB classifier predictor",
                short_description="Predict the class labels using Bernoulli naive Bayes (BNB) classifier")
class BernoulliNaiveBayesClassifierPredictor(BaseSupervisedPredictor):
    """
    Predictor of a Naive Bayes classifier. Perform classification on a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html for more details
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        BernoulliNaiveBayesClassifierResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
