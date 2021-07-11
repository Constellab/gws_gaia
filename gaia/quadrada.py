# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from gws.process import Process
from gws.resource import Resource

from .data import Tuple
from .dataset import Dataset

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, qda: QuadraticDiscriminantAnalysis = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['qda'] = qda

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of quadratic discriminant analysis model. Fit a quadratic discriminant analysis model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'reg_param': {"type": 'float', "default": 0},
    }

    async def task(self):
        dataset = self.input['dataset']
        qda = QuadraticDiscriminantAnalysis(reg_param=self.get_param("reg_param"))
        qda.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(qda=qda)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained quadratic discriminant analysis model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained quadratic discriminant analysis model.
    
    See ttps://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        qda = learned_model.kv_store['qda']
        y = qda.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of quadratic discriminant analysis model. Predic class labels of a dataset with a trained quadratic discriminant analysis model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        qda = learned_model.kv_store['qda']
        y = qda.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset