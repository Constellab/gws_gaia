# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

from gws_core import (Process, Resource, ProcessDecorator, ResourceDecorator)
from ..data.core import Tuple
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@ResourceDecorator("RandomForestClassifierResult", hide=True)
class RandomForestClassifierResult(Resource):
    def __init__(self, rfc: RandomForestClassifier = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['rfc'] = rfc

#==============================================================================
#==============================================================================

@ProcessDecorator("RandomForestClassifierTrainer")
class RandomForestClassifierTrainer(Process):
    """
    Trainer of a random forest classifier. Build a forest of trees from a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : RandomForestClassifierResult}
    config_specs = {
        'nb_estimators': {"type": 'int', "default": 100, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        rfc = RandomForestClassifier(n_estimators=self.get_param("nb_estimators"))
        rfc.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(rfc=rfc)
        self.output['result'] = result

#==============================================================================
#==============================================================================

@ProcessDecorator("RandomForestClassifierTester")
class RandomForestClassifierTester(Process):
    """
    Tester of a trained random forest classifier. Return the mean accuracy on a given dataset for a trained random forest classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': RandomForestClassifierResult}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        rfc = learned_model.kv_store['rfc']
        y = rfc.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

@ProcessDecorator("RandomForestClassifierPredictor")
class RandomForestClassifierPredictor(Process):
    """
    Predictor of a random forest classifier. Predict class labels of a dataset with a trained random forest classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': RandomForestClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        rfc = learned_model.kv_store['rfc']
        y = rfc.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset