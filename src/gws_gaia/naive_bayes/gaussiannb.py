# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB

from gws_core import (Task, Resource, task_decorator, resource_decorator)

from ..data.core import Tuple
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@resource_decorator("GaussianNaiveBayesResult", hide=True)
class GaussianNaiveBayesResult(Resource):
    def __init__(self, gnb: GaussianNB = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['gnb'] = gnb

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

    async def task(self):
        dataset = self.input['dataset']
        gnb = GaussianNB()
        gnb.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(gnb=gnb)
        self.output['result'] = result

#==============================================================================
#==============================================================================

@task_decorator("GaussianNaiveBayesTester")
class GaussianNaiveBayesTester(Task):
    """
    Tester of a trained gaussian Naïve Bayes model. Return the mean accuracy on a given dataset for a trained gaussian Naïve Bayes model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GaussianNaiveBayesResult}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        gnb = learned_model.kv_store['gnb']
        y = gnb.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

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
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        gnb = learned_model.kv_store['gnb']
        y = gnb.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset