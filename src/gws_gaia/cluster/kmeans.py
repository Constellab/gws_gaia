# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from pandas import DataFrame
from sklearn.cluster import KMeans

from gws_core import (Task, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("KMeansResult")
class KMeansResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("KMeansTrainer")
class KMeansTrainer(Task):
    """
    Trainer of a trained k-means clustering model. Compute a k-means clustering from a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : KMeansResult}
    config_specs = {
        'nb_clusters': IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        kmeans = KMeans(n_clusters=params["nb_clusters"])
        kmeans.fit(dataset.features.values)
        result = KMeansResult(result = kmeans)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("KMeansPredictor")
class KMeansPredictor(Task):
    """
    Predictor of a K-means clustering model. Predict the closest cluster each sample in a dataset belongs to.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': KMeansResult}
    output_specs = {'result' : Dataset}
    config_specs = { }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        kmeans = learned_model.result
        y = kmeans.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}