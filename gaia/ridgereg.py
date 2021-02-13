# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource

from sklearn.linear_model import Ridge
from gaia.data import Tuple
from numpy import ravel

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, rir: Ridge = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['rir'] = rir

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a Ridge regression model. Fit a Ridge regression model with a training dataset. 

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'alpha':{"type": 'float', "default": 1, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        rir = Ridge(alpha=self.get_param("alpha"))
        rir.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(rir=rir)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained Ridge regression model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained Ridge regression model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        rir = learned_model.kv_store['rir']
        y = rir.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a Ridge regression model. Predict target values of a dataset with a trained Ridge regression model. 

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        rir = learned_model.kv_store['rir']
        y = rir.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset