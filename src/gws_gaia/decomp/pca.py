# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (BarPlotView, ConfigParams, Dataset, FloatParam,
                      FloatRField, IntParam, Resource, ResourceRField,
                      ScatterPlot2DView, ScatterPlot3DView, StrParam, Table,
                      TableView, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, view)
from pandas import DataFrame
from sklearn.decomposition import PCA

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# PCATrainerResult
#
# *****************************************************************************


@resource_decorator("PCATrainerResult", hide=True)
class PCATrainerResult(BaseResource):

    _training_set: Resource = ResourceRField()  # pour lier ressources entre elles
    _log_likelihood: int = FloatRField()  # list, float, dict,...

    def _get_transformed_data(self) -> DataFrame:  # retourne DataFrame
        pca: PCA = self.get_result()  # typage de pca du type PCA
        ncomp = pca.n_components_
        x_transformed: DataFrame = pca.transform(self._training_set.get_features().values)
        columns = [f"PC{n+1}" for n in range(0, ncomp)]
        x_transformed = DataFrame(data=x_transformed, columns=columns, index=self._training_set.instance_names)
        return x_transformed

    def _get_log_likelihood(self) -> float:
        if not self._log_likelihood:
            pca = self.get_result()
            self._log_likelihood = pca.score(self._training_set.get_features().values)
        return self._log_likelihood

    @view(view_type=TableView, human_name="Projected data table",
          short_description="Table of data projected in the score plot")
    def view_transformed_data_as_table(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        x_transformed = self._get_transformed_data()
        table = Table(x_transformed)
        return TableView(table)

    @view(view_type=TableView, human_name="Variance table", short_description="Table of explained variances")
    def view_variance_as_table(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        pca = self.get_result()
        index = [f"PC{n+1}" for n in range(0, pca.n_components_)]
        columns = ["ExplainedVariance"]
        data = DataFrame(pca.explained_variance_ratio_, columns=columns, index=index)
        table = Table(data)
        return TableView(table)

    @view(view_type=BarPlotView, human_name="Variance bar plot", short_description="Barplot of explained variances")
    def view_variance_as_barplot(self, params: ConfigParams) -> dict:
        """
        View bar plot of explained variances
        """

        pca = self.get_result()
        explained_var: DataFrame = pca.explained_variance_ratio_
        columns = [f"PC{n+1}" for n in range(0, pca.n_components_)]
        _view = BarPlotView()
        _view.add_series(
            x=columns,
            y=explained_var.tolist()
        )
        _view.x_label = 'Principal components'
        _view.y_label = 'Explained variance'

        return _view

    @view(view_type=ScatterPlot2DView, human_name='2D-score plot', short_description='2D score plot')
    def view_scores_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        data: DataFrame = self._get_transformed_data()
        _view = ScatterPlot2DView()
        _view.add_series(
            x=data['PC1'].to_list(),
            y=data['PC2'].to_list()
        )
        _view.x_label = 'PC1'
        _view.y_label = 'PC2'
        return _view

    # @view(view_type=ScatterPlot3DView, human_name='3D-score plot', short_description='3D-score plot')
    # def view_scores_as_3d_plot(self, params: ConfigParams) -> dict:
    #     """
    #     View 3D score plot
    #     """

    #     data: DataFrame = self._get_transformed_data()
    #     _view = ScatterPlot2DView()
    #     view.add_series(
    #         x=data['PC1'].to_list(),
    #         y=data['PC2'].to_list(),
    #         z=data['PC3'].to_list()
    #     )
    #     _view.x_label = 'PC1'
    #     _view.y_label = 'PC2'
    #     _view.z_label = 'PC3'
    #     return _view

# *****************************************************************************
#
# PCATrainer
#
# *****************************************************************************


@task_decorator("PCATrainer")
class PCATrainer(Task):
    """
    Trainer of a Principal Component Analysis (PCA) model. Fit a PCA model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html for more details
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': PCATrainerResult}
    config_specs = {
        'nb_components': IntParam(default_value=2, min_value=2)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        ncomp = params["nb_components"]
        pca = PCA(n_components=ncomp)
        print(dataset.get_features().values)
        pca.fit(dataset.get_features().values)
        result = PCATrainerResult(result=pca)
        result._training_set = dataset
        return {'result': result}

# *****************************************************************************
#
# PCATransformer
#
# *****************************************************************************


@task_decorator("PCATransformer")
class PCATransformer(Task):
    """
    Transformer of a Principal Component Analysis (PCA) model. Apply dimensionality reduction to a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html for more details

    """
    input_specs = {'dataset': Dataset, 'learned_model': PCATrainerResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        pca = learned_model.get_result()
        x_transformed = pca.transform(dataset.get_features().values)
        ncomp = x_transformed.shape[1]
        result = Dataset(
            data=x_transformed,
            row_names=dataset.row_names,
            column_names=["PC"+str(i+1) for i in range(0, ncomp)]
        )
        return {'result': result}
