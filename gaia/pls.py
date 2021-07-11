# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from pandas import DataFrame
from sklearn.cross_decomposition import PLSRegression

from gws.process import Process
from gws.resource import Resource

from .data import Tuple
from .dataset import Dataset

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, pls: PLSRegression = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['pls'] = pls

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a Partial Least Squares (PLS) regression model. Fit a PLS regression model to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'nb_components': {"type": 'int', "default": 2, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        pls = PLSRegression(n_components=self.get_param("nb_components"))
        pls.fit(dataset.features.values, dataset.targets.values)
        
        t = self.output_specs["result"]
        result = t(pls=pls)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained Partial Least Squares (PLS) regression model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained Partial Least Squares (PLS) regression model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        pls = learned_model.kv_store['pls']
        y = pls.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a Partial Least Squares (PLS) regression model. Predict targets of a dataset with a trained PLS regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        pls = learned_model.kv_store['pls']
        y = pls.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset