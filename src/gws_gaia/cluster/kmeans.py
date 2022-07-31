# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (ConfigParams, Dataset, FloatParam, FloatRField,
                      InputSpec, IntParam, OutputSpec, Resource,
                      ResourceRField, ScatterPlot2DView, ScatterPlot3DView,
                      StrParam, Table, TabularView, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, view)
from numpy import concatenate, ndarray, transpose, unique, vstack
from pandas import DataFrame, concat
from sklearn.cluster import KMeans

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# KMeansResult
#
# *****************************************************************************


@resource_decorator("KMeansResult", hide=True)
class KMeansResult(BaseResourceSet):
    """ KMeansResult """
    LABELED_TABLE_NAME = "Labeled data table"

    def __init__(self, training_set=None, result=None):
        super().__init__(training_set=training_set, result=result)
        # append tables
        if training_set is not None:
            self._create_labeled_table()

    def _create_labeled_table(self):
        kmeans: KMeans = self.get_result()
        columns = self.get_training_set().feature_names
#        columns.extend(['label'])
        train_set = self.get_training_set().get_features().values
        label = kmeans.labels_[:, None]
        label = label.tolist()
        flat_label = list(concatenate(label).flat)
        label_name = ['label'] * len(label)
        # -----------------
        # Interleaving flat_label and label_name
        res = label_name + flat_label
        res[::2] = label_name
        res[1::2] = flat_label
        label = [res[i:i+2] for i in range(0, len(res), 2)]
        # -----------------
        for i in range(len(label)):
            label[i] = {label[i][0]: label[i][1]}
        # for i, lbl in enumerate(label):
        #    label[i] = {lbl[0]: lbl[1]}
        data = train_set
        data = DataFrame(data, index=self.get_training_set().row_names, columns=columns)
        table = Table(data=data)
        row_tags = label
        table.name = self.LABELED_TABLE_NAME
        table.set_all_rows_tags(row_tags)
        self.add_resource(table)

    @view(view_type=TabularView, human_name="Label table", short_description="Table of labels")
    def view_labels_as_table(self, params: ConfigParams) -> dict:
        """
        View Table
        """
        kmeans = self.get_result()
        columns = self.get_training_set().feature_names
        columns.extend(['label'])
        train_set = self.get_training_set().get_features().values
        label = kmeans.labels_[:, None]
        data = concatenate((train_set, label), axis=1)
        data = DataFrame(data, index=self.get_training_set().row_names, columns=columns)
        t_view = TabularView()
        t_view.set_data(data=data)
        return t_view

    @view(view_type=ScatterPlot2DView, human_name='2D-score plot', short_description='2D-score plot')
    def view_labels_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View 2D scatter plot
        """

        kmeans = self.get_result()
        columns = self.get_training_set().feature_names
        train_set = self.get_training_set().get_features().values
        label = kmeans.labels_[:, None]
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

# *****************************************************************************
#
# KMeansTrainer
#
# *****************************************************************************


@task_decorator("KMeansTrainer", human_name="KMeans trainer", short_description="Train a K-Means clustering model")
class KMeansTrainer(Task):
    """
    Trainer of a trained k-means clustering model. Compute a k-means clustering from a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(KMeansResult, human_name="result", short_description="The output result")}
    config_specs = {
        'nb_clusters': IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        kmeans = KMeans(n_clusters=params["nb_clusters"])
        kmeans.fit(dataset.get_features().values)
        result = KMeansResult(training_set=dataset, result=kmeans)
        return {'result': result}

# *****************************************************************************
#
# KMeansPredictor
#
# *****************************************************************************


@task_decorator("KMeansPredictor", human_name="KMeans predictor",
                short_description="Predict the closest cluster each sample using a K-Means model")
class KMeansPredictor(Task):
    """
    Predictor of a K-means clustering model. Predict the closest cluster of each sample in a dataset belongs to.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more details.
    """
    input_specs = {
        'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
        'learned_model': InputSpec(KMeansResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        kmeans = learned_model.get_result()
        y = kmeans.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=DataFrame(y),
            row_names=dataset.row_names,
            column_names=["label"],
            target_names=["label"],
        )
        return {'result': result_dataset}
