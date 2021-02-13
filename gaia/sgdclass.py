# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource

from sklearn.linear_model import SGDClassifier
from gaia.data import Tuple
from numpy import ravel

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, sgdc: SGDClassifier = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['sgdc'] = sgdc

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a linear classifier with stochastic gradient descent (SGD). Fit a SGD linear classifier with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'loss': {"type": 'str', "default": 'hinge'},
        'alpha': {"type": 'float', "default": 0.0001, "min": 0},
        'max_iter': {"type": 'int', "default": 1000, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        sgdc = SGDClassifier(max_iter=self.get_param("max_iter"),alpha=self.get_param("alpha"),loss=self.get_param("loss"))
        sgdc.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(sgdc=sgdc)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained linear classifier with stochastic gradient descent (SGD). Return the mean accuracy on a given dataset for a trained linear classifier with stochastic gradient descent (SGD).
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        sgdc = learned_model.kv_store['sgdc']
        y = sgdc.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a linear classifier with stochastic gradient descent (SGD). Predict class labels of a dataset with a trained SGD linear classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html for more details.
    """    
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        sgdc = learned_model.kv_store['sgdc']
        y = sgdc.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset