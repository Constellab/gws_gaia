# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from pandas import DataFrame
from sklearn.cluster import KMeans

from gws_core import (Process, Resource, ProcessDecorator, ResourceDecorator)
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@ResourceDecorator("KMeansResult")
class KMeansResult(Resource):
    def __init__(self, kmeans: KMeans = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['kmeans'] = kmeans

#==============================================================================
#==============================================================================

@ProcessDecorator("KMeansTrainer")
class KMeansTrainer(Process):
    """
    Trainer of a trained k-means clustering model. Compute a k-means clustering from a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : KMeansResult}
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

@ProcessDecorator("KMeansPredictor")
class KMeansPredictor(Process):
    """
    Predictor of a K-means clustering model. Predict the closest cluster each sample in a dataset belongs to.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': KMeansResult}
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