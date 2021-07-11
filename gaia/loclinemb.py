# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from sklearn.manifold import LocallyLinearEmbedding

from gws.process import Process
from gws.resource import Resource

from .dataset import Dataset

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, lle: LocallyLinearEmbedding = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['lle'] = lle

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a locally linear embedding model. Compute the embedding vectors for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'nb_components': {"type": 'int', "default": 2, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        lle = LocallyLinearEmbedding(n_components=self.get_param("nb_components"))
        lle.fit(dataset.features.values)

        t = self.output_specs["result"]
        result = t(lle=lle)
        self.output['result'] = result