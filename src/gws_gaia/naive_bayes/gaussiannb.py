# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("GaussianNaiveBayesResult", hide=True)
class GaussianNaiveBayesResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("GaussianNaiveBayesTrainer")
class GaussianNaiveBayesTrainer(Task):
    """
    Trainer of a gaussian naive Bayes model. Fit a gaussian naive Bayes according to a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : GaussianNaiveBayesResult}
    config_specs = {

    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        gnb = GaussianNB()
        gnb.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = GaussianNaiveBayesResult(result=gnb)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("GaussianNaiveBayesPredictor")
class GaussianNaiveBayesPredictor(Task):
    """
    Predictor of a gaussian naïve Bayes model. Perform classification on a dataset.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GaussianNaiveBayesResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gnb = learned_model.result
        y = gnb.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}