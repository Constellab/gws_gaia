# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource

from sklearn.kernel_ridge import KernelRidge
from gaia.data import Tuple
from numpy import ravel

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, krr: KernelRidge = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['krr'] = krr

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a kernel ridge regression model. Fit a kernel ridge regression model with a training dataset. 

    See https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'gamma': {"type": 'float', "default": None},
        'kernel': {"type": 'str', "default": 'linear'}
    }

    async def task(self):
        dataset = self.input['dataset']
        krr = KernelRidge(gamma=self.get_param("gamma"),kernel=self.get_param("kernel"))
        krr.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(krr=krr)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained kernel ridge regression model. Return the coefficient of determination R^2 of the prediction on a given dataset for a kernel ridge regression model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        krr = learned_model.kv_store['krr']
        y = krr.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a kernel ridge regression model. Predict a regression target from a dataset with a trained kernel ridge regression model. 

    See https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        krr = learned_model.kv_store['krr']
        y = krr.predict(dataset.features.values)
        
        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset