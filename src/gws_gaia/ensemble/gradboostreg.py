# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingRegressor

from gws_core import (Process, Resource, ProcessDecorator, ResourceDecorator)
from ..data.core import Tuple
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================


@ResourceDecorator("GradientBoostingRegressorResult", hide=True)
class GradientBoostingRegressorResult(Resource):
    def __init__(self, gbr: GradientBoostingRegressor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['gbr'] = gbr

#==============================================================================
#==============================================================================

@ProcessDecorator("GradientBoostingRegressorTrainer")
class GradientBoostingRegressorTrainer(Process):
    """
    Trainer of a gradient boosting regressor. Fit a gradient boosting regressor with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : GradientBoostingRegressorResult}
    config_specs = {
        'nb_estimators': {"type": 'int', "default": 100, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        gbr = GradientBoostingRegressor(n_estimators=self.get_param("nb_estimators"))
        gbr.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(gbr=gbr)
        self.output['result'] = result

#==============================================================================
#==============================================================================

@ProcessDecorator("GradientBoostingRegressorTester")
class GradientBoostingRegressorTester(Process):
    """
    Tester of a trained gradient boosting regressor. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained gradient boosting regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GradientBoostingRegressorResult}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        gbr = learned_model.kv_store['gbr']
        y = gbr.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

@ProcessDecorator("GradientBoostingRegressorPredictor")
class GradientBoostingRegressorPredictor(Process):
    """
    Predictor of a gradient boosting regressor. Predict regression target for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GradientBoostingRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        gbr = learned_model.kv_store['gbr']
        y = gbr.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset