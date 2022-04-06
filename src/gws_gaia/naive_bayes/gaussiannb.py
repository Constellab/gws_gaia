# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator)
from numpy import ravel
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# GaussianNaiveBayesResult
#
# *****************************************************************************


@resource_decorator("GaussianNaiveBayesResult", hide=True)
class GaussianNaiveBayesResult(BaseResource):
    pass

# *****************************************************************************
#
# GaussianNaiveBayesTrainer
#
# *****************************************************************************


@task_decorator("GaussianNaiveBayesTrainer", human_name="GNB classifier trainer",
                short_description="Train a Gaussian naive Bayes (GNB) classifier")
class GaussianNaiveBayesTrainer(Task):
    """
    Trainer of a gaussian naive Bayes model. Fit a gaussian naive Bayes according to a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html for more details
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': GaussianNaiveBayesResult}
    config_specs = {

    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        gnb = GaussianNB()
        gnb.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = GaussianNaiveBayesResult(training_set=dataset, result=gnb)
        return {'result': result}

# *****************************************************************************
#
# GaussianNaiveBayesPredictor
#
# *****************************************************************************


@task_decorator("GaussianNaiveBayesPredictor", human_name="GNB predictor",
                short_description="Predict the class labels using Gaussian naive Bayes (GNB) classifier")
class GaussianNaiveBayesPredictor(Task):
    """
    Predictor of a gaussian naive Bayes model. Perform classification on a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html for more details
    """
    input_specs = {'dataset': Dataset, 'learned_model': GaussianNaiveBayesResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gnb = learned_model.get_result()
        y = gnb.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
