# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator)
from numpy import ravel
from pandas import DataFrame
from sklearn.naive_bayes import BernoulliNB

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# BernoulliNaiveBayesClassifierResult
#
# *****************************************************************************


@resource_decorator("BernoulliNaiveBayesClassifierResult", hide=True)
class BernoulliNaiveBayesClassifierResult(BaseResource):
    pass

# *****************************************************************************
#
# BernoulliNaiveBayesClassifierTrainer
#
# *****************************************************************************


@task_decorator("BernoulliNaiveBayesClassifierTrainer", human_name="BNB classifier trainer",
                short_description="Train a Bernoulli naive Bayes (BNB) classifier model")
class BernoulliNaiveBayesClassifierTrainer(Task):
    """
    Trainer of a Naive Bayes classifier. Fit Naive Bayes classifier with dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html for more details
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': BernoulliNaiveBayesClassifierResult}
    config_specs = {
        'alpha': FloatParam(default_value=1)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        bnb = BernoulliNB(alpha=params["alpha"])
        bnb.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = BernoulliNaiveBayesClassifierResult(training_set=dataset, result=bnb)
        return {'result': result}

# *****************************************************************************
#
# BernoulliNaiveBayesClassifierPredictor
#
# *****************************************************************************


@task_decorator("BernoulliNaiveBayesClassifierPredictor", human_name="BNB classifier predictor",
                short_description="Predict the class labels using Bernoulli naive Bayes (BNB) classifier")
class BernoulliNaiveBayesClassifierPredictor(Task):
    """
    Predictor of a Naive Bayes classifier. Perform classification on a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html for more details
    """
    input_specs = {'dataset': Dataset, 'learned_model': BernoulliNaiveBayesClassifierResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        bnb = learned_model.get_result()
        y = bnb.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
