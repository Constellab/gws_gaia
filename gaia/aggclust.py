# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset

from gws.model import Config
from gws.model import Process, Config, Resource

from sklearn.cluster import AgglomerativeClustering

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, aggclust: AgglomerativeClustering = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.kv_store['pls'] = pls

#==============================================================================
#==============================================================================

class Trainer(Process):
    """ Trainer of the hierarchical clustering. Fits the hierarchical clustering from features, or distance matrix.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'nb_clusters': {"type": 'int', "default": 2, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        aggclust = AgglomerativeClustering(n_clusters=self.get_param("nb_clusters"))
        aggclust.fit(dataset.features.values)
        
        t = self.output_specs["result"]
        result = t(aggclust=aggclust)
        self.output['result'] = result