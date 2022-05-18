# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (BadRequestException, ConfigParams, Dataset, IntParam,
                      Resource, ScatterPlot2DView, StrParam, Table, Task,
                      TaskInputs, TaskOutputs, resource_decorator,
                      task_decorator, view, InputSpec, OutputSpec)
from pandas import DataFrame
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import OrdinationResults
from skbio.stats.ordination import pcoa as PCoA

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# PCATrainerResult
#
# *****************************************************************************


@resource_decorator("PCoATrainerResult", human_name="PCoA trainer results",
                    short_description="Principal Coordinate Analysis result", hide=True)
class PCoATrainerResult(BaseResourceSet):
    """
    PCoATrainerResult
    """

    TRANSFORMED_TABLE_NAME = "Transformed data table"
    VARIANCE_TABLE_NAME = "Variance table"

    def __init__(self, training_set=None, result=None):
        super().__init__(training_set=training_set, result=result)
        # append tables
        if training_set is not None:
            self._create_transformed_table()
            self._create_variance_table()

    def _create_transformed_table(self):
        pcoa: OrdinationResults = self.get_result()
        x_transformed = pcoa.samples
        ncomp = x_transformed.shape[1]
        columns = [f"PC{n+1}" for n in range(0, ncomp)]
        data = DataFrame(data=x_transformed, columns=columns, index=self.get_training_set().instance_names)
        table = Table(data=data)
        row_tags = self.get_training_set().get_row_tags()
        table.name = self.TRANSFORMED_TABLE_NAME
        table.set_row_tags(row_tags)
        self.add_resource(table)

    def _create_variance_table(self):
        pcoa: OrdinationResults = self.get_result()

        explained_variance_ratio = pcoa.proportion_explained
        ncomp = explained_variance_ratio.shape[0]

        index = [f"PC{n+1}" for n in range(0, ncomp)]
        columns = ["ExplainedVariance"]
        data = DataFrame(explained_variance_ratio, columns=columns, index=index)
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

    @view(view_type=ScatterPlot2DView, human_name='2D-score plot', short_description='2D score plot')
    def view_scores_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        data: DataFrame = self.get_transformed_table().get_data()
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
# PCoATrainer
#
# *****************************************************************************


@ task_decorator("PCoATrainer", human_name="PCoA trainer",
                 short_description="Train a Principal Coordinate Analysis (PCoA) model")
class PCoATrainer(Task):
    """
    Trainer of a Principal Coordinate Analysis (PCoA) model. Fit a PCoA model with a training dataset.

    See http://scikit-bio.org/docs/0.5.7/generated/skbio.stats.ordination.pcoa.html for more details
    """
    input_specs = {'distance_table': InputSpec((Dataset, Table), human_name="Dataset", short_description="The input distance table")}
    output_specs = {'result': OutputSpec(PCoATrainerResult, human_name="result", short_description="The output result")}
    config_specs = {
        'nb_components': IntParam(default_value=2, min_value=2),
        'method': StrParam(default_value='eigh', allowed_values=['eigh', 'fsvd'])
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        distance_table = inputs['distance_table']
        distance_matrix = distance_table.get_data()

        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise BadRequestException(
                "Invalid distance table. The number of rows must be equal to the number of columns.")
        ncomp = params["nb_components"]
        method = params["method"]
        distance_matrix = DistanceMatrix(
            distance_matrix.to_numpy(),
            ids=distance_matrix.index,
            validate=True
        )
        pcoa: PCoA = PCoA(distance_matrix, method=method, number_of_dimensions=ncomp, inplace=True)
        result = PCoATrainerResult(training_set=distance_table, result=pcoa)
        return {'result': result}
