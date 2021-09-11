# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.naive_bayes import BernoulliNB

from gws_core import (Task, Resource, task_decorator, resource_decorator)

from ..data.core import Tuple
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@resource_decorator("BernoulliNaiveBayesClassifierResult", hide=True)
class BernoulliNaiveBayesClassifierResult(Resource):
    def __init__(self, bnb: BernoulliNB = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['bnb'] = bnb

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
        'alpha':{"type": 'float', "default": 1}
    }

    async def task(self):
        dataset = self.input['dataset']
        bnb = BernoulliNB(alpha=self.get_param("alpha"))
        bnb.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(bnb=bnb)
        self.output['result'] = result

#==============================================================================
#==============================================================================

@task_decorator("BernoulliNaiveBayesClassifierTester")
class BernoulliNaiveBayesClassifierTester(Task):
    """
    Tester of a trained Naive Bayes classifier. Return the mean accuracy on a given test data and labels for a trained Naive Bayes classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': BernoulliNaiveBayesClassifierResult}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        bnb = learned_model.kv_store['bnb']
        y = bnb.score(dataset.features.values,ravel(dataset.targets.values))
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

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
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        bnb = learned_model.kv_store['bnb']
        y = bnb.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset