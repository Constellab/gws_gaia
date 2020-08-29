# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset

from gws.prism.model import Config
from gws.prism.controller import Controller
from gws.prism.model import Process, Config, Resource

from sklearn.decomposition import PCA


class Result(Resource):
    def __init__(self, pca: PCA = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.store['pca'] = pca

class Trainer(Process):
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'nb_components': {"type": 'int', "default": 2, "min": 0}
    }

    def task(self):
        dataset = self.input['dataset']
        pca = PCA(n_components=self.get_param("nb_components"))
        pca.fit(dataset.features.values)
        result = Result(pca=pca)
        self.output['result'] = result