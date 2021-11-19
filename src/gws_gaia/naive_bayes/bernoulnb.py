# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.naive_bayes import BernoulliNB

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("BernoulliNaiveBayesClassifierResult", hide=True)
class BernoulliNaiveBayesClassifierResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("BernoulliNaiveBayesClassifierTrainer")
class BernoulliNaiveBayesClassifierTrainer(Task):
    """
    Trainer of a Naive Bayes classifier. Fit Naive Bayes classifier with dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : BernoulliNaiveBayesClassifierResult}
    config_specs = {
        'alpha': FloatParam(default_value=1)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        bnb = BernoulliNB(alpha=params["alpha"])
        bnb.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = BernoulliNaiveBayesClassifierResult(result = bnb)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("BernoulliNaiveBayesClassifierPredictor")
class BernoulliNaiveBayesClassifierPredictor(Task):
    """
    Predictor of a Naive Bayes classifier. Perform classification on a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': BernoulliNaiveBayesClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        bnb = learned_model.result
        y = bnb.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}