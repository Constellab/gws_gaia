# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from numpy import ravel
from pandas import DataFrame
from sklearn.tree import DecisionTreeRegressor

from gws.process import Process
from gws.resource import Resource

from .data import Tuple
from .dataset import Dataset


#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, dtr: DecisionTreeRegressor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['dtr'] = dtr

#==============================================================================
#==============================================================================

class Trainer(Process):
    """ Trainer of a decision tree regressor. Build a decision tree regressor from a training set   

    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'max_depth': {"type": 'int', "default": None, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        dtr = DecisionTreeRegressor(max_depth=self.get_param("max_depth"))
        dtr.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(dtr=dtr)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained decision tree regressor. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained decision tree regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        dtr = learned_model.kv_store['dtr']
        y = dtr.score(dataset.features.values,ravel(dataset.targets.values))
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """ Predictor of a trained decision tree regressor. Predict class or regression value for the dataset. For a classification model, the predicted class for each sample in the dataset is returned. For a regression model, the predicted value based on the dataset is returned.

    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        dtr = learned_model.kv_store['dtr']
        y = dtr.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset