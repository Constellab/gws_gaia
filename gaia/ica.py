# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from sklearn.decomposition import FastICA

from gws.process import Process
from gws.resource import Resource

from .dataset import Dataset

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, ica: FastICA = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['ica'] = ica

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of an Independant Component Analysis (ICA). Fit a model of ICA to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA.fit for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'nb_components': {"type": 'int', "default": 2, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        ica = FastICA(n_components=self.get_param("nb_components"))
        ica.fit(dataset.features.values)

        t = self.output_specs["result"]
        result = t(ica=ica)
        self.output['result'] = result