# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import SGDRegressor

from gws_core import (Process, Resource, ProcessDecorator, ResourceDecorator)

from ..data.dataset import Dataset
from ..data.core import Tuple

#==============================================================================
#==============================================================================

@ResourceDecorator("SGDRegressorResult", hide=True)
class SGDRegressorResult(Resource):
    def __init__(self, sgdr: SGDRegressor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['sgdr'] = sgdr

#==============================================================================
#==============================================================================

@ProcessDecorator("SGDRegressorTrainer")
class SGDRegressorTrainer(Process):
    """
    Trainer of a linear regressor with stochastic gradient descent (SGD). Fit a SGD linear regressor with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for more details.
    """    
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : SGDRegressorResult}
    config_specs = {
        'loss': {"type": 'str', "default": 'squared_loss'},
        'alpha': {"type": 'float', "default": 0.0001, "min": 0},
        'max_iter': {"type": 'int', "default": 1000, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        sgdr = SGDRegressor(max_iter=self.get_param("max_iter"),alpha=self.get_param("alpha"),loss=self.get_param("loss"))
        sgdr.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(sgdr=sgdr)
        self.output['result'] = result

#==============================================================================
#==============================================================================

@ProcessDecorator("SGDRegressorTester")
class SGDRegressorTester(Process):
    """
    Tester of a trained linear regressor with stochastic gradient descent (SGD). Return the coefficient of determination R^2 of the prediction on a given dataset for a trained linear regressor with stochastic gradient descent (SGD).
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': SGDRegressorResult}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        sgdr = learned_model.kv_store['sgdr']
        y = sgdr.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

@ProcessDecorator("SGDRegressorPredictor")
class SGDRegressorPredictor(Process):
    """
    Predictor of a linear regressor with stochastic gradient descent (SGD). Predict target values of a dataset with a trained SGD linear regressor.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for more details.
    """        
    input_specs = {'dataset' : Dataset, 'learned_model': SGDRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        sgdr = learned_model.kv_store['sgdr']
        y = sgdr.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset