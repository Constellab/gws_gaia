# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from pandas import DataFrame, concat
from numpy import concatenate, transpose, ndarray, vstack
from sklearn.cluster import KMeans

from gws_core import (Task, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam,
                        ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, 
                        FloatRField, Resource, Table)
from gws_core import Dataset
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# KMeansResult
#
# *****************************************************************************

@resource_decorator("KMeansResult")
class KMeansResult(BaseResource):

    _training_set: Resource = ResourceRField()
    
    @view(view_type=TableView, human_name="LabelsTable", short_description="Table of labels")
    def view_labels_as_table(self, params: ConfigParams) -> dict:
        """
        View Table
        """
        kmeans = self.get_result()
        columns = self._training_set.feature_names
        columns.extend(['label'])
        train_set = self._training_set.get_features().values
        label = kmeans.labels_[:, None]
        data = concatenate((train_set, label), axis=1)       
        table = Table(data, column_names=columns, row_names=self._training_set.row_names)
        return TableView(table)

# *****************************************************************************
#
# KMeansTrainer
#
# *****************************************************************************

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
        kmeans.fit(dataset.get_features().values)
        result = KMeansResult(result = kmeans)
        result._training_set = dataset
        return {'result': result}

# *****************************************************************************
#
# KMeansPredictor
#
# *****************************************************************************

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
        y = kmeans.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data = DataFrame(y),
            row_names = dataset.row_names,
            column_names = ["label"],
            target_names = ["label"],
        )
        return {'result': result_dataset}