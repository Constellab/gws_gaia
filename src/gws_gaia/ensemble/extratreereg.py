# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import ExtraTreesRegressor

from gws_core import (Process, Resource, ProcessDecorator, ResourceDecorator)
from ..data.core import Tuple
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@ResourceDecorator("ExtraTreesRegressorResult", hide=True)
class ExtraTreesRegressorResult(Resource):
    def __init__(self, etr: ExtraTreesRegressor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['etr'] = etr

#==============================================================================
#==============================================================================

@ProcessDecorator("ExtraTreesRegressorTrainer")
class ExtraTreesRegressorTrainer(Process):
    """
    Trainer of an extra-trees regressor. Build a forest of trees from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : ExtraTreesRegressorResult}
    config_specs = {
        'nb_estimators': {"type": 'int', "default": 100, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        etr = ExtraTreesRegressor(n_estimators=self.get_param("nb_estimators"))
        etr.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(etr=etr)
        self.output['result'] = result

#==============================================================================
#==============================================================================

@ProcessDecorator("ExtraTreesRegressorTester")
class ExtraTreesRegressorTester(Process):
    """
    Tester of a trained extra-trees regressor. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained extra-trees regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': ExtraTreesRegressorResult}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        etr = learned_model.kv_store['etr']
        y = etr.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

@ProcessDecorator("ExtraTreesRegressorPredictor")
class ExtraTreesRegressorPredictor(Process):
    """
    Predictor of an extra-trees regressor. Predict regression target of a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': ExtraTreesRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        etr = learned_model.kv_store['etr']
        y = etr.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset