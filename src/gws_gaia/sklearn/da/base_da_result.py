# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, ScatterPlot2DView, Table,
                      resource_decorator, view)
from pandas import DataFrame

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import BaseSupervisedResult

# *****************************************************************************
#
# LDAResult
#
# *****************************************************************************


@ resource_decorator("BaseDAResult", hide=True)
class BaseDAResult(BaseSupervisedResult):

    TRANSFORMED_TABLE_NAME = "Transformed table"
    VARIANCE_TABLE_NAME = "Variance table"

    def __init__(self, training_set=None, training_design=None, result=None):
        super().__init__(training_set=training_set, training_design=training_design, result=result)
        if training_set is not None:
            self._create_transformed_table()
            self._create_variance_table()

    def _create_transformed_table(self) -> DataFrame:
        mdl = self.get_result()
        ncomp = mdl.explained_variance_ratio_.shape[0]

        training_set = self.get_training_set()
        training_design = self.get_training_design()
        x_true, _ = TrainingDesignHelper.create_training_matrices(training_set, training_design)

        data: DataFrame = mdl.transform(x_true.values)
        columns = [f"PC{i+1}" for i in range(0, ncomp)]
        data = DataFrame(
            data=data,
            columns=columns,
            index=training_set.row_names
        )
        table = Table(data=data)
        row_tags = training_set.get_row_tags()
        table.name = self.TRANSFORMED_TABLE_NAME
        table.set_all_row_tags(row_tags)
        self.add_resource(table)

    def _create_variance_table(self):
        mdl = self.get_result()
        ncomp = mdl.explained_variance_ratio_.shape[0]
        index = [f"PC{i+1}" for i in range(0, ncomp)]
        columns = ["ExplainedVariance"]
        data = DataFrame(mdl.explained_variance_ratio_, index=index, columns=columns)
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

    @ view(view_type=ScatterPlot2DView, human_name='2D-score plot', short_description='2D-score plot')
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
