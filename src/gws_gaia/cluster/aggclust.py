# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from pandas import DataFrame, concat
from numpy import concatenate, transpose, ndarray, vstack, unique
from typing import List
from sklearn.cluster import AgglomerativeClustering

from gws_core import (Task, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam,
                        ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, 
                        FloatRField, Resource, Table)
from gws_core import Dataset
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# AgglomerativeClusteringResult
#
# *****************************************************************************

@resource_decorator("AgglomerativeClusteringResult", hide=True)
class AgglomerativeClusteringResult(BaseResource):

    _training_set: Resource = ResourceRField()
    
    @view(view_type=TableView, human_name="LabelsTable", short_description="Table of labels")
    def view_labels_as_table(self, params: ConfigParams) -> dict:
        """
        View Table
        """
        aggclust = self.get_result()
        columns = self._training_set.feature_names
        columns.extend(['label'])
        train_set = self._training_set.get_features().values
        label = aggclust.labels_[:, None]
        data = concatenate((train_set, label), axis=1)       
        table = Table(data, column_names=columns, row_names=self._training_set.row_names)
        return TableView(table)

    @view(view_type=ScatterPlot2DView, human_name='ScorePlot2D', short_description='2D score plot')
    def view_labels_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View 2D scatter plot
        """

        aggclust = self.get_result()
        columns = self._training_set.feature_names
        train_set = self._training_set.get_features().values
        label = aggclust.labels_[:, None]
        label_values = unique(label)
        flatten_label = label.flatten()
        
        _view = ScatterPlot2DView()        
        for l in label_values:
            data = train_set[flatten_label==l]   
            _view.add_series(
                x = data[:,0].tolist(),
                y = data[:,1].tolist()
            )
        
        _view.x_label = columns[0]
        _view.y_label = columns[1]
        return _view

    def view_as_dendrogram(self, params: ConfigParams):
        pass

    #def get_labels(self) -> List[str]:
    #    return self.get_result().labels_
    
    #def view_as_table(self, params: ConfigParams):
    #    pass

# *****************************************************************************
#
# AgglomerativeClusteringTrainer
#
# *****************************************************************************

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
        result._training_set = dataset
        return {'result': result}
