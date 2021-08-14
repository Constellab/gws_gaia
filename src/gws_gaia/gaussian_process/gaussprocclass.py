# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.gaussian_process import GaussianProcessClassifier

from gws_core import (Process, Resource, ProcessDecorator, ResourceDecorator)

from ..data.core import Tuple
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@ResourceDecorator("GaussianProcessClassifierResult", hide=True)
class GaussianProcessClassifierResult(Resource):
    def __init__(self, gpc: GaussianProcessClassifier = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['gpc'] = gpc

#==============================================================================
#==============================================================================

@ProcessDecorator("GaussianProcessClassifierTrainer")
class GaussianProcessClassifierTrainer(Process):
    """
    Trainer of a Gaussian process classifier. Fit a Gaussian process classification model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier.fit for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : GaussianProcessClassifierResult}
    config_specs = {
        'random_state': {"type": 'int', "default": None, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        gpc = GaussianProcessClassifier(random_state=self.get_param("random_state"))
        gpc.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(gpc=gpc)
        self.output['result'] = result

#==============================================================================
#==============================================================================

@ProcessDecorator("GaussianProcessClassifierTester")
class GaussianProcessClassifierTester(Process):
    """
    Tester of a trained Gaussian process classifier. Return the mean accuracy on a given dataset for a trained Gaussian process classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GaussianProcessClassifierResult}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        gpc = learned_model.kv_store['gpc']
        y = gpc.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

@ProcessDecorator("GaussianProcessClassifierPredictor")
class GaussianProcessClassifierPredictor(Process):
    """
    Predictor of a Gaussian process classifier. Predict classes of a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier.fit for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': GaussianProcessClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        gpc = learned_model.kv_store['gpc']
        y = gpc.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset