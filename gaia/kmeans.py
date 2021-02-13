# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource

from sklearn.cluster import KMeans

#==============================================================================
#==============================================================================

class Result(Resource):
    def __init__(self, kmeans: KMeans = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['kmeans'] = kmeans

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a trained K-means clustering model. Compute a k-means clustering from a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'nb_clusters': {"type": 'int', "default": 2, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        kmeans = KMeans(n_clusters=self.get_param("nb_clusters"))
        kmeans.fit(dataset.features.values)

        t = self.output_specs["result"]
        result = t(kmeans=kmeans)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a K-means clustering model. Predict the closest cluster each sample in a dataset belongs to.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        kmeans = learned_model.kv_store['kmeans']
        y = kmeans.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset