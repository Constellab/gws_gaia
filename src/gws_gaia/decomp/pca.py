# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (BarPlotView, BoolParam, ConfigParams, Dataset,
                      FloatRField, IntParam, Resource, ResourceRField,
                      ScatterPlot2DView, ScatterPlot3DView, Table, TabularView,
                      Task, TaskInputs, TaskOutputs, resource_decorator,
                      task_decorator, view)
from pandas import DataFrame
from sklearn.decomposition import PCA

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# PCATrainerResult
#
# *****************************************************************************


@resource_decorator("PCATrainerResult", hide=True)
class PCATrainerResult(BaseResourceSet):

    _log_likelihood: int = FloatRField()

    TRANSFORMED_TABLE_NAME = "Transformed data table"
    VARIANCE_TABLE_NAME = "Variance table"

    def __init__(self, training_set=None, result=None):
        super().__init__(training_set=training_set, result=result)
        # append tables
        if training_set is not None:
            self._create_transformed_table()
            self._create_variance_table()

    def _create_transformed_table(self):
        pca: PCA = self.get_result()  # typage de pca du type PCA
        ncomp = pca.n_components_
        x_transformed: DataFrame = pca.transform(self._training_set.get_features().values)
        columns = [f"PC{n+1}" for n in range(0, ncomp)]
        data = DataFrame(data=x_transformed, columns=columns, index=self._training_set.instance_names)
        table = Table(data=data)
        row_tags = self._training_set.get_row_tags()
        table.name = self.TRANSFORMED_TABLE_NAME
        table.set_row_tags(row_tags)
        self.add_resource(table)

    def _create_variance_table(self):
        pca = self.get_result()
        index = [f"PC{n+1}" for n in range(0, pca.n_components_)]
        columns = ["ExplainedVariance"]
        data = DataFrame(pca.explained_variance_ratio_, columns=columns, index=index)
        table = Table(data=data)
        table.name = self.VARIANCE_TABLE_NAME
        self.add_resource(table)

    def get_transformed_table(self):
        if self.resource_exists(self.TRANSFORMED_TABLE_NAME):
            return self.get_resource(self.TRANSFORMED_TABLE_NAME)
        else:
            return None

    def get_variance_table(self):
        if self.resource_exists(self.VARIANCE_TABLE_NAME):
            return self.get_resource(self.VARIANCE_TABLE_NAME)
        else:
            return None

    def _get_log_likelihood(self) -> float:
        if not self._log_likelihood:
            pca = self.get_result()
            self._log_likelihood = pca.score(self._training_set.get_features().values)
        return self._log_likelihood

    @view(view_type=ScatterPlot2DView, default_view=True, human_name='2D-score plot', short_description='2D score plot')
    def view_scores_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        data: DataFrame = self.get_transformed_table().get_data()
        _view = ScatterPlot2DView()
        row_tags = self._training_set.get_row_tags()
        _view.add_series(
            x=data['PC1'].to_list(),
            y=data['PC2'].to_list(),
            tags=row_tags
        )
        _view.x_label = 'PC1'
        _view.y_label = 'PC2'
        return _view

# *****************************************************************************
#
# PCATrainer
#
# *****************************************************************************


@ task_decorator("PCATrainer", human_name="PCA trainer",
                 short_description="Train a Principal Component Analysis (PCA) model")
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
        pca.fit(dataset.get_features().values)
        result = PCATrainerResult(training_set=dataset, result=pca)
        return {'result': result}

# *****************************************************************************
#
# PCATransformer
#
# *****************************************************************************


@ task_decorator("PCATransformer", human_name="PCA transformer",
                 short_description="Transform a data using a Principal Component Analysis (PCA) model. Apply dimensionality reduction to a dataset")
class PCATransformer(Task):
    """
    Transformer using Principal Component Analysis (PCA) model. Apply dimensionality reduction to a dataset

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
