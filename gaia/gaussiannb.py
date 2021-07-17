# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB

from gws.process import Process
from gws.resource import Resource

from .data import Tuple
from .dataset import Dataset

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, gnb: GaussianNB = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['gnb'] = gnb

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a gaussian naive Bayes model. Fit a gaussian naive Bayes according to a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
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

class Tester(Process):
    """
    Tester of a trained gaussian Naïve Bayes model. Return the mean accuracy on a given dataset for a trained gaussian Naïve Bayes model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
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

class Predictor(Process):
    """
    Predictor of a gaussian naïve Bayes model. Perform classification on a dataset.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
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