# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource

from sklearn.mixture import GaussianMixture
from numpy import ravel

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, gmixt: GaussianMixture = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['gmixt'] = gmixt

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a Gaussian mixture model. Estimate model parameters with a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'nb_components': {"type": 'int', "default": 1, "min": 0},
        'covariance_type': {"type": 'str', "default": 'full'}
    }

    async def task(self):
        dataset = self.input['dataset']
        gmixt = GaussianMixture(n_components=self.get_param("nb_components"),covariance_type=self.get_param("covariance_type"))
        gmixt.fit(dataset.features.values)
        
        t = self.output_specs["result"]
        result = t(gmixt=gmixt)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a Gaussian mixture model. Predict the labels for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        gmixt = learned_model.kv_store['gmixt']
        y = gmixt.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset