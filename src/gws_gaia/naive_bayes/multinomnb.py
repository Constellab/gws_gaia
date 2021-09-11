# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB

from gws_core import (Task, Resource, task_decorator, resource_decorator)

from ..data.core import Tuple
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@resource_decorator("MultinomialNaiveBayesClassifierResult", hide=True)
class MultinomialNaiveBayesClassifierResult(Resource):
    def __init__(self, mnb: MultinomialNB = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['mnb'] = mnb

#==============================================================================
#==============================================================================

@task_decorator("MultinomialNaiveBayesClassifierTrainer")
class MultinomialNaiveBayesClassifierTrainer(Task):
    """
    Trainer of a naive Bayes classifier for a multinomial model. Fit a naive Bayes classifier according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : MultinomialNaiveBayesClassifierResult}
    config_specs = {
        'alpha':{"type": 'float', "default": 1}
    }

    async def task(self):
        dataset = self.input['dataset']
        mnb = MultinomialNB(alpha=self.get_param("alpha"))
        mnb.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(mnb=mnb)
        self.output['result'] = result

#==============================================================================
#==============================================================================

@task_decorator("MultinomialNaiveBayesClassifierTester")
class MultinomialNaiveBayesClassifierTester(Task):
    """
    Tester of a naïve Bayes classifier for a multinomial model. Return the mean accuracy on a given dataset for a trained naïve Bayes classifier for a multinomial model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': MultinomialNaiveBayesClassifierResult}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        mnb = learned_model.kv_store['mnb']
        y = mnb.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

@task_decorator("MultinomialNaiveBayesClassifierPredictor")
class MultinomialNaiveBayesClassifierPredictor(Task):
    """
    Predictor of a naïve Bayes classifier for a multinomial model. Predict class labels for a dataset using a trained naïve Bayes classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': MultinomialNaiveBayesClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        mnb = learned_model.kv_store['mnb']
        y = mnb.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset