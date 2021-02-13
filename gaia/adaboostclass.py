# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource

from sklearn.ensemble import AdaBoostClassifier
from gaia.data import Tuple
from numpy import ravel

class Result(Resource):
    def __init__(self, *args, abc: AdaBoostClassifier = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['abc'] = abc

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of an Adaboost classifier. This process builds a boosted classifier from a training set.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'nb_estimators': {"type": 'int', "default": 50, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        abc = AdaBoostClassifier(n_estimators=self.get_param("nb_estimators"))
        abc.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(abc=abc)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained Adaboost classifier. Return the mean accuracy on a given test data and labels for a trained Adaboost classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        abc = learned_model.kv_store['abc']
        y = abc.score(dataset.features.values,ravel(dataset.targets.values))
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a trained Adaboost classifier. This process predicts classes for a dataset.
    The predicted class of an input sample is computed as the weighted mean prediction of the classifiers in the ensemble.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        abc = learned_model.kv_store['abc']
        y = abc.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset