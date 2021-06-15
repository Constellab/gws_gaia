# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.model import Process, Config, Resource

from sklearn.neighbors import KNeighborsRegressor
from gaia.data import Tuple
from numpy import ravel

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, neigh: KNeighborsRegressor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['neigh'] = neigh

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer for a k-nearest neighbors regressor. Fit a k-nearest neighbors regressor from a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'nb_neighbors': {"type": 'int', "default": 5, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        neigh = KNeighborsRegressor(n_neighbors=self.get_param("nb_neighbors"))
        neigh.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(neigh=neigh)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained k-nearest neighbors regressor. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained k-nearest neighbors regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        neigh = learned_model.kv_store['neigh']
        y = neigh.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor for a k-nearest neighbors regressor. Predict the regression target for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        neigh = learned_model.kv_store['neigh']
        y = neigh.predict(dataset.features.values)
        
        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset