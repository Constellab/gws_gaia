# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource

from sklearn.decomposition import PCA

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, pca: PCA = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['pca'] = pca

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a Principal Component Analysis (PCA) model. Fit a PCA model with a training dataset.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'nb_components': {"type": 'int', "default": 2, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        pca = PCA(n_components=self.get_param("nb_components"))
        pca.fit(dataset.features.values)

        t = self.output_specs["result"]
        result = t(pca=pca)
        #a = result.kv_store['pca']
        self.output['result'] = result