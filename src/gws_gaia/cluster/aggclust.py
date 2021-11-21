# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import List
from sklearn.cluster import AgglomerativeClustering

from gws_core import (Task, Resource, 
                        task_decorator, resource_decorator, 
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam, view)
from gws_core import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("AgglomerativeClusteringResult", hide=True)
class AgglomerativeClusteringResult(BaseResource):
    
    def get_labels(self) -> List[str]:
        return self.get_result().labels_
    
    def view_as_dendrogram(self, params: ConfigParams):
        pass

    def view_as_table(self, params: ConfigParams):
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
        "nb_clusters": IntParam(default_value=2, min_value=0),
        "linkage": StrParam(default_value="ward", allowed_values=["ward", "complete", "average", "single"]),
        "affinity": StrParam(default_value="euclidean", allowed_values=["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"], short_description="Metric used to compute the linkage."),
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        aggclust = AgglomerativeClustering(n_clusters=params["nb_clusters"], linkage=params["linkage"])
        aggclust.fit(dataset.get_features().values)
        result = AgglomerativeClusteringResult(result = aggclust)
        return {'result': result}
