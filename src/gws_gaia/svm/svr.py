# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.svm import SVR

from gws_core import Process, Resource, ResourceDecorator, ProcessDecorator
from ..data.core import Tuple
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@ResourceDecorator("SVRResult", hide=True)
class SVRResult(Resource):
    def __init__(self, svr: SVR = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['svr'] = svr

#==============================================================================
#==============================================================================

@ProcessDecorator("SVRTrainer")
class SVRTrainer(Process):
    """
    Trainer of a Epsilon-Support Vector Regression (SVR) model. Fit a SVR model according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : SVRResult}
    config_specs = {
        'kernel': {"type": 'str', "default": 'rbf'}
    }

    async def task(self):
        dataset = self.input['dataset']
        svr = SVR(kernel=self.get_param("kernel"))
        svr.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(svr=svr)
        self.output['result'] = result

#==============================================================================
#==============================================================================

@ProcessDecorator("SVRTester")
class SVRTester(Process):
    """
    Tester of a trained Epsilon-Support Vector Regression (SVR) model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained Epsilon-Support Vector Regression (SVR) model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': SVRResult}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        svr = learned_model.kv_store['svr']
        y = svr.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

@ProcessDecorator("SVRPredictor")
class SVRPredictor(Process):
    """
    Predictor of a Epsilon-Support Vector Regression (SVR) model. Predict target values of a dataset with a trained SVR model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html for more details.
    """    
    input_specs = {'dataset' : Dataset, 'learned_model': SVRResult}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        svr = learned_model.kv_store['svr']
        y = svr.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset