# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.model import Process, Config, Resource

from sklearn.svm import SVC
from gaia.data import Tuple
from numpy import ravel

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, svc: SVC = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['svc'] = svc

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a C-Support Vector Classifier (SVC) model. Fit a SVC model according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'probability': {"type": 'bool', "default": False},
        'kernel': {"type": 'str', "default": 'rbf'}
    }

    async def task(self):
        dataset = self.input['dataset']
        svc = SVC(probability=self.get_param("probability"),kernel=self.get_param("kernel"))
        svc.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(svc=svc)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained C-Support Vector Classifier (SVC) model. Return the mean accuracy on a given dataset for a trained C-Support Vector Classifier (SVC) model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        svc = learned_model.kv_store['svc']
        y = svc.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a C-Support Vector Classifier (SVC) model. Predict class labels of a dataset with a trained SVC model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html for more details.
    """    
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        svc = learned_model.kv_store['svc']
        y = svc.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset