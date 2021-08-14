# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier

from gws_core import (Process, Resource, ProcessDecorator, ResourceDecorator)
from ..data.core import Tuple
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@ResourceDecorator("GradientBoostingClassifierResult", hide=True)
class GradientBoostingClassifierResult(Resource):
    def __init__(self, gbc: GradientBoostingClassifier = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['gbc'] = gbc

#==============================================================================
#==============================================================================

@ProcessDecorator("GradientBoostingClassifierTrainer")
class GradientBoostingClassifierTrainer(Process):
    """
    Trainer of a gradient boosting classifier. Fit a gradient boosting classifier with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : GradientBoostingClassifierResult}
    config_specs = {
        'nb_estimators': {"type": 'int', "default": 100, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        gbc = GradientBoostingClassifier(n_estimators=self.get_param("nb_estimators"))
        gbc.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(gbc=gbc)
        self.output['result'] = result

#==============================================================================
#==============================================================================

@ProcessDecorator("GradientBoostingClassifierTester")
class GradientBoostingClassifierTester(Process):
    """
    Tester of a trained gradient boosting classifier. Return the mean accuracy on a given dataset for a trained gradient boosting classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GradientBoostingClassifierResult}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        gbc = learned_model.kv_store['gbc']
        y = gbc.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

@ProcessDecorator("GradientBoostingClassifierPredictor")
class GradientBoostingClassifierPredictor(Process):
    """
    Predictor of a gradient boosting classifier. Predict classes for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GradientBoostingClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        gbc = learned_model.kv_store['gbc']
        y = gbc.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset