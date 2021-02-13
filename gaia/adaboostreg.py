# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource

from sklearn.ensemble import AdaBoostRegressor
from gaia.data import Tuple
from numpy import ravel

class Result(Resource):
    def __init__(self, abr: AdaBoostRegressor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['abr'] = abr

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of an Adaboost regressor. This process build a boosted regressor from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'nb_estimators': {"type": 'int', "default": 50, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        abr = AdaBoostRegressor(n_estimators=self.get_param("nb_estimators"))
        abr.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(abr=abr)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained Adaboost regressor. Return the coefficient of determination R^2 of the prediction on a given test data for a trained Adaboost regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        abr = learned_model.kv_store['abr']
        y = abr.score(dataset.features.values,ravel(dataset.targets.values))
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a trained Adaboost regressor. The predicted regression value of an input sample is computed as the weighted median 
    prediction of the classifiers in the ensemble.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        abr = learned_model.kv_store['abr']
        y = abr.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset