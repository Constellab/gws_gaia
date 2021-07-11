# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import Lasso

from gws.process import Process
from gws.resource import Resource

from .data import Tuple
from .dataset import Dataset

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, las: Lasso = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['las'] = las

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a lasso model. Fit a lasso model with a training dataset.

    See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.Lasso.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'alpha':{"type": 'float', "default": 1, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        las = Lasso(alpha=self.get_param("alpha"))
        las.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(las=las)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained lasso model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained lasso model.
    
    See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.Lasso.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        las = learned_model.kv_store['las']
        y = las.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a lasso model. Predict target values from a dataset with a trained lasso model.

    See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.Lasso.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        las = learned_model.kv_store['las']
        y = las.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset