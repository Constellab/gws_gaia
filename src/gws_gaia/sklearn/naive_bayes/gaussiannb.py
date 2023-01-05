# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, OutputSpec, Table, resource_decorator,
                      task_decorator)
from sklearn.naive_bayes import GaussianNB

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedClassResult,
                             BaseSupervisedPredictor, BaseSupervisedTrainer)

# *****************************************************************************
#
# GaussianNaiveBayesResult
#
# *****************************************************************************


@resource_decorator("GaussianNaiveBayesResult", hide=True)
class GaussianNaiveBayesResult(BaseSupervisedClassResult):
    pass

# *****************************************************************************
#
# GaussianNaiveBayesTrainer
#
# *****************************************************************************


@task_decorator("GaussianNaiveBayesTrainer", human_name="GNB classifier trainer",
                short_description="Train a Gaussian naive Bayes (GNB) classifier")
class GaussianNaiveBayesTrainer(BaseSupervisedTrainer):
    """
    Trainer of a gaussian naive Bayes model. Fit a gaussian naive Bayes according to a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(GaussianNaiveBayesResult, human_name="result",
                                         short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return GaussianNB()

    @classmethod
    def create_result_class(cls) -> Type[GaussianNaiveBayesResult]:
        return GaussianNaiveBayesResult

# *****************************************************************************
#
# GaussianNaiveBayesPredictor
#
# *****************************************************************************


@task_decorator("GaussianNaiveBayesPredictor", human_name="GNB predictor",
                short_description="Predict the class labels using Gaussian naive Bayes (GNB) classifier")
class GaussianNaiveBayesPredictor(BaseSupervisedPredictor):
    """
    Predictor of a gaussian naive Bayes model. Perform classification on a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        GaussianNaiveBayesResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
