# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.ensemble import ExtraTreesClassifier

from gws_core import (Process, Resource, ProcessDecorator, ResourceDecorator)
from ..data.core import Tuple
from ..data.dataset import Dataset


#==============================================================================
#==============================================================================

@ResourceDecorator("ExtraTreesClassifierResult", hide=True)
class ExtraTreesClassifierResult(Resource):
    def __init__(self, etc: ExtraTreesClassifier = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['etc'] = etc

#==============================================================================
#==============================================================================

@ProcessDecorator("ExtraTreesClassifierTrainer")
class ExtraTreesClassifierTrainer(Process):
    """
    Trainer of an extra-trees classifier. Build a forest of trees from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier.fit for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : ExtraTreesClassifierResult}
    config_specs = {
        'nb_estimators': {"type": 'int', "default": 100, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        etc = ExtraTreesClassifier(n_estimators=self.get_param("nb_estimators"))
        etc.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(etc=etc)
        self.output['result'] = result

#==============================================================================
#==============================================================================

@ProcessDecorator("ExtraTreesClassifierTester")
class ExtraTreesClassifierTester(Process):
    """
    Tester of a trained extra-trees classifier. Return the mean accuracy on a given dataset for a trained extra-trees classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': ExtraTreesClassifierResult}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        etc = learned_model.kv_store['etc']
        y = etc.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

@ProcessDecorator("ExtraTreesClassifierPredictor")
class ExtraTreesClassifierPredictor(Process):
    """
    Predictor of an extra-trees classifier. Predict class for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier.fit for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': ExtraTreesClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        etc = learned_model.kv_store['etc']
        y = etc.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset