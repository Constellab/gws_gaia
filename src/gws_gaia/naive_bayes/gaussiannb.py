# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("GaussianNaiveBayesResult", hide=True)
class GaussianNaiveBayesResult(Resource):
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
        gnb.fit(dataset.features.values, ravel(dataset.targets.values))
        result = GaussianNaiveBayesResult.from_result(gnb=gnb)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("GaussianNaiveBayesTester")
class GaussianNaiveBayesTester(Task):
    """
    Tester of a trained gaussian Naïve Bayes model. Return the mean accuracy on a given dataset for a trained gaussian Naïve Bayes model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GaussianNaiveBayesResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        gnb = learned_model.binary_store['result']
        y = gnb.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = GenericResult.from_result(result = z)
        return {'result': result_dataset}

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
        gnb = learned_model.binary_store['result']
        y = gnb.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}