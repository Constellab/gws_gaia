# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import List

from gws_core import (ConfigParams, Dataset, FloatParam, FloatRField,
                      InputSpec, IntParam, OutputSpec, Resource,
                      ResourceRField, ScatterPlot2DView, ScatterPlot3DView,
                      StrParam, Table, TableView, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, view)
from numpy import concatenate, ndarray, transpose, unique, vstack
from pandas import DataFrame, concat
from sklearn.cluster import AgglomerativeClustering

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# AgglomerativeClusteringResult
#
# *****************************************************************************


@resource_decorator("AgglomerativeClusteringResult", hide=True)
class AgglomerativeClusteringResult(BaseResourceSet):
    """ AgglomerativeClusteringResult """

    @view(view_type=TableView, human_name="Label table", short_description="Table of labels")
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
        data = DataFrame(data, index=self._training_set.row_names, columns=columns)
        t_view = TableView(Table(data))
        return t_view

    @view(view_type=ScatterPlot2DView, human_name='2D-score plot', short_description='2D-score plot')
    def view_labels_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View 2D scatter plot
        """

        aggclust = self.get_result()
        columns = self._training_set.row_names
        train_set = self._training_set.get_features().values
        label = aggclust.labels_[:, None]
        label_values = unique(label)
        flatten_label = label.flatten()

        _view = ScatterPlot2DView()
        for l in label_values:
            data = train_set[flatten_label == l]
            _view.add_series(
                x=data[:, 0].tolist(),
                y=data[:, 1].tolist()
            )

        _view.x_label = columns[0]
        _view.y_label = columns[1]
        return _view

    def view_as_dendrogram(self, params: ConfigParams):
        pass

# *****************************************************************************
#
# AgglomerativeClusteringTrainer
#
# *****************************************************************************


@task_decorator("AgglomerativeClusteringTrainer", human_name="Agglomerative clustering trainer",
                short_description="Train a the hierarchical clustering model")
class AgglomerativeClusteringTrainer(Task):
    """ Trainer of the hierarchical clustering. Fits the hierarchical clustering from features, or distance matrix.
@
    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html for more details
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(AgglomerativeClusteringResult,
                                         human_name="result", short_description="The output result")}
    config_specs = {
        "nb_clusters": IntParam(default_value=2, min_value=0),
        "linkage": StrParam(default_value="ward", allowed_values=["ward", "complete", "average", "single"]),
        "affinity":
        StrParam(
            default_value="euclidean", allowed_values=["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
            short_description="Metric used to compute the linkage."), }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        aggclust = AgglomerativeClustering(n_clusters=params["nb_clusters"], linkage=params["linkage"])
        aggclust.fit(dataset.get_features().values)
        result = AgglomerativeClusteringResult(training_set=dataset, result=aggclust)
        return {'result': result}
