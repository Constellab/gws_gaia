# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import LinearRegression

from gws_core import (Process, Resource, ProcessDecorator, ResourceDecorator)

from ..data.core import Tuple
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@ResourceDecorator("LinearRegressionResult", hide=True)
class LinearRegressionResult(Resource):
    def __init__(self, lir: LinearRegression = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['lir'] = lir

#==============================================================================
#==============================================================================

@ProcessDecorator("LinearRegressionTrainer")
class LinearRegressionTrainer(Process):
    """
    Trainer fo a linear regression model. Fit a linear regression model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : LinearRegressionResult}
    config_specs = {
    }

    async def task(self):
        dataset = self.input['dataset']
        lir = LinearRegression()
        lir.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(lir=lir)
        self.output['result'] = result

#==============================================================================
#==============================================================================

@ProcessDecorator("LinearRegressionTester")
class LinearRegressionTester(Process):
    """
    Tester of a trained linear regression model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained linear regression model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LinearRegressionResult}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        lir = learned_model.kv_store['lir']
        y = lir.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

@ProcessDecorator("LinearRegressionPredictor")
class LinearRegressionPredictor(Process):
    """
    Predictor of a linear regression model. Predict target values of a dataset with a trained linear regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LinearRegressionResult}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        lir = learned_model.kv_store['lir']
        y = lir.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset