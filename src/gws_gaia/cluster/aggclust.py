# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from sklearn.cluster import AgglomerativeClustering

from gws_core import (Task, Resource, 
                        task_decorator, resource_decorator, 
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam, view)
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("AgglomerativeClusteringResult", hide=True)
class AgglomerativeClusteringResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("AgglomerativeClusteringTrainer")
class AgglomerativeClusteringTrainer(Task):
    """ Trainer of the hierarchical clustering. Fits the hierarchical clustering from features, or distance matrix.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : AgglomerativeClusteringResult}
    config_specs = {
        'nb_clusters': IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        aggclust = AgglomerativeClustering(n_clusters=params["nb_clusters"])
        aggclust.fit(dataset.get_features().values)
        result = AgglomerativeClusteringResult(result = aggclust)
        return {'result': result}
