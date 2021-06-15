# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.model import Process, Config, Resource

from sklearn.ensemble import RandomForestRegressor
from gaia.data import Tuple
from numpy import ravel

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, rfr: RandomForestRegressor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['rfr'] = rfr

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a random forest regressor. Build a forest of trees from a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'nb_estimators': {"type": 'int', "default": 100, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        rfr = RandomForestRegressor(n_estimators=self.get_param("nb_estimators"))
        rfr.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(rfr=rfr)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained random forest regressor. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained random forest regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        rfr = learned_model.kv_store['rfr']
        y = rfr.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a random forest regressor. Predict regression target of a dataset with a trained random forest regressor.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        rfr = learned_model.kv_store['rfr']
        y = rfr.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset