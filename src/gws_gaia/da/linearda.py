# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (BarPlotView, ConfigParams, Dataset, FloatParam,
                      FloatRField, IntParam, IntRField, Resource,
                      ResourceRField, ScatterPlot2DView, ScatterPlot3DView,
                      StrParam, Table, TabularView, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, view, InputSpec, OutputSpec)
from numpy import ravel, shape, unique
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# LDAResult
#
# *****************************************************************************


@resource_decorator("LDAResult",
                    human_name="LDA result",
                    short_description="Linear Discriminant Analysis result",
                    hide=True)
class LDAResult(BaseResourceSet):

    TRANSFORMED_TABLE_NAME = "Transformed data table"
    VARIANCE_TABLE_NAME = "Variance table"

    def __init__(self, training_set=None, result=None):
        super().__init__(training_set=training_set, result=result)
        # append tables
        if training_set is not None:
            self._create_transformed_table()
            self._create_variance_table()

    def _create_transformed_table(self) -> DataFrame:
        lda: LinearDiscriminantAnalysis = self.get_result()
        ncomp = lda.explained_variance_ratio_.shape[0]
        data: DataFrame = lda.transform(self.get_training_set().get_features().values)
        columns = [f"PC{i+1}" for i in range(0, ncomp)]
        data = DataFrame(
            data=data,
            columns=columns,
            index=self.get_training_set().row_names
        )
        table = Table(data=data)
        row_tags = self.get_training_set().get_row_tags()
        table.name = self.TRANSFORMED_TABLE_NAME
        table.set_row_tags(row_tags)
        self.add_resource(table)

    def _create_variance_table(self):
        lda = self.get_result()
        ncomp = lda.explained_variance_ratio_.shape[0]
        index = [f"PC{i+1}" for i in range(0, ncomp)]
        columns = ["ExplainedVariance"]
        data = DataFrame(lda.explained_variance_ratio_, index=index, columns=columns)
        table = Table(data=data)
        table.name = self.VARIANCE_TABLE_NAME
        self.add_resource(table)

    def get_transformed_table(self):
        """ Get transformed table """
        if self.resource_exists(self.TRANSFORMED_TABLE_NAME):
            return self.get_resource(self.TRANSFORMED_TABLE_NAME)
        else:
            return None

    def get_variance_table(self):
        """ Get variance table """

        if self.resource_exists(self.VARIANCE_TABLE_NAME):
            return self.get_resource(self.VARIANCE_TABLE_NAME)
        else:
            return None

    @view(view_type=ScatterPlot2DView, human_name='2D-score plot', short_description='2D-score plot')
    def view_scores_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        data = self._get_transformed_data()
        _view = ScatterPlot2DView()
        row_tags = self.get_training_set().get_row_tags()
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
# LDATrainer
#
# *****************************************************************************


@task_decorator("LDATrainer", human_name="LDA trainer",
                short_description="Train a linear discriminant analysis classifier")
class LDATrainer(Task):
    """
    Trainer of a linear discriminant analysis classifier. Fit Linear Discriminant Analysis model according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(LDAResult, human_name="result", short_description="The output result")}
    config_specs = {
        'solver': StrParam(default_value='svd'),
        'nb_components': IntParam(default_value=None, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        lda = LinearDiscriminantAnalysis(solver=params["solver"], n_components=params["nb_components"])
        lda.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = LDAResult(training_set=dataset, result=lda)
        return {'result': result}

# *****************************************************************************
#
# LDATransformer
#
# *****************************************************************************


@task_decorator("LDATransformer", human_name="LDA transformer",
                short_description="Transform a dataset using of a Linear Discriminant Analysis (LDA) classifier")
class LDATransformer(Task):
    """
    Transformer of a linear discriminant analysis classifier. Project data to maximize class separation.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details

    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(LDAResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lda = learned_model.get_result()
        x = lda.transform(dataset.get_features().values)
        ncomp = x.shape[1]
        result_dataset = Dataset(
            data=DataFrame(x),
            row_names=dataset.row_names,
            column_names=[f"PC{i+1}" for i in range(0, ncomp)]
        )
        return {'result': result_dataset}

# *****************************************************************************
#
# LDAPredictor
#
# *****************************************************************************


@task_decorator("LDAPredictor", human_name="LDA predictor",
                short_description="Predict class labels using a Linear Discriminant Analysis (LDA) classifier")
class LDAPredictor(Task):
    """
    Predictor of a linear discriminant analysis classifier. Predict class labels for samples in a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(LDAResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lda = learned_model.get_result()
        y = lda.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=DataFrame(y),
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names,
        )
        return {'result': result_dataset}
