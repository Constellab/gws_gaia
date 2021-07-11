# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import ElasticNet

from gws.process import Process
from gws.resource import Resource

from .data import Tuple
from .dataset import Dataset

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, *args, eln: ElasticNet = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['eln'] = eln

#==============================================================================
#==============================================================================

class Trainer(Process):
    """ 
    Trainer of an elastic net model. Fit model with coordinate descent.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'alpha':{"type": 'float', "default": 1, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        eln = ElasticNet(alpha=self.get_param("alpha"))
        eln.fit(dataset.features.values, dataset.targets.values)
        
        t = self.output_specs["result"]
        result = t(eln=eln)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained elastic net model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained elastic net model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        eln = learned_model.kv_store['eln']
        y = eln.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a trained elastic net model. Predict from a dataset using the trained model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        eln = learned_model.kv_store['eln']
        y = eln.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset