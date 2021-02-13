# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource

from sklearn.gaussian_process import GaussianProcessRegressor
from gaia.data import Tuple
from numpy import ravel

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, gpr: GaussianProcessRegressor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['gpr'] = gpr

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a Gaussian process regressor. Fit a Gaussian process regressor model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html for more details
    """    
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'alpha': {"type": 'float', "default": 1e-10, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        gpr = GaussianProcessRegressor(alpha=self.get_param("alpha"))
        gpr.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(gpr=gpr)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained Gaussian process regressor. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained Gaussian process regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        gpr = learned_model.kv_store['gpr']
        y = gpr.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a Gaussian process regressor. Predict regression targets of a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html for more details
    """    
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        gpr = learned_model.kv_store['gpr']
        y = gpr.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset