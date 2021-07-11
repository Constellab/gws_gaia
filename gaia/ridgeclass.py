# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import RidgeClassifier

from gws.process import Process
from gws.resource import Resource

from .data import Tuple
from .dataset import Dataset

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, ric: RidgeClassifier = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['ric'] = ric

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a Ridge regression classifier. Fit a Ridge classifier model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'alpha':{"type": 'float', "default": 1, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        ric = RidgeClassifier(alpha=self.get_param("alpha"))
        ric.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(ric=ric)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained Ridge regression classifier. Return the mean accuracy on a given dataset for a trained Ridge regression classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        ric = learned_model.kv_store['ric']
        y = ric.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a Ridge regression classifier. Predict class labels for samples in a datatset with a trained Ridge classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for more details.
    """    
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        ric = learned_model.kv_store['ric']
        y = ric.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset