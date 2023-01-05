# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (ConfigParams, FloatRField, InputSpec, IntParam,
                      OutputSpec, ScatterPlot2DView, Table, TechnicalInfo,
                      resource_decorator, task_decorator, view)
from pandas import DataFrame
from sklearn.decomposition import PCA

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_unsup import BaseUnsupervisedResult, BaseUnsupervisedTrainer

# *****************************************************************************
#
# PCATrainerResult
#
# *****************************************************************************


@resource_decorator("PCATrainerResult", hide=True)
class PCATrainerResult(BaseUnsupervisedResult):
    """PCATrainerResult  """

    TRANSFORMED_TABLE_NAME = "Transformed table"
    VARIANCE_TABLE_NAME = "Variance table"
    _log_likelihood: int = FloatRField()

    def __init__(self, training_set=None, training_design=None, result=None):
        super().__init__(training_set=training_set, training_design=training_design, result=result)
        if training_set is not None:
            self._create_transformed_table()
            self._create_variance_table()
            self._create_log_likelihood()

    def _create_transformed_table(self):
        pca: PCA = self.get_result()  # typage de pca du type PCA
        ncomp = pca.n_components_
        data: DataFrame = pca.transform(self.get_training_set().get_data())
        columns = [f"PC{n+1}" for n in range(0, ncomp)]
        data = DataFrame(data=data, columns=columns, index=self.get_training_set().row_names)
        table = Table(data=data)
        row_tags = self.get_training_set().get_row_tags()
        table.name = self.TRANSFORMED_TABLE_NAME
        table.set_all_row_tags(row_tags)
        self.add_resource(table)

    def _create_variance_table(self):
        pca = self.get_result()
        index = [f"PC{n+1}" for n in range(0, pca.n_components_)]
        columns = ["ExplainedVariance"]
        data = DataFrame(pca.explained_variance_ratio_, columns=columns, index=index)
        table = Table(data=data)
        table.name = self.VARIANCE_TABLE_NAME
        self.add_resource(table)
        self.add_technical_info(TechnicalInfo(key='PC1', value=f'{data.iat[0,0]:.3f}'))
        self.add_technical_info(TechnicalInfo(key='PC2', value=f'{data.iat[1,0]:.3f}'))

    def _create_log_likelihood(self) -> float:
        if not self._log_likelihood:
            mdl = self.get_result()
            training_set = self.get_training_set()
            training_design = self.get_training_design()
            x_true, y_true = TrainingDesignHelper.create_training_matrices(training_set, training_design)
            self._log_likelihood = mdl.score(
                X=x_true,
                y=y_true
            )
        technical_info = TechnicalInfo(key='Log likelihood', value=self._log_likelihood)
        self.add_technical_info(technical_info)

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
        var = self.get_variance_table().get_data()
        _view.x_label = f'PC1 ({100*var.iat[0,0]:.2f}%)'
        _view.y_label = f'PC2 ({100*var.iat[1,0]:.2f}%)'
        return _view

# *****************************************************************************
#
# PCATrainer
#
# *****************************************************************************


@ task_decorator("PCATrainer", human_name="PCA trainer",
                 short_description="Train a Principal Component Analysis (PCA) model")
class PCATrainer(BaseUnsupervisedTrainer):
    """
    Trainer of a Principal Component Analysis (PCA) model. Fit a PCA model with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(PCATrainerResult, human_name="result", short_description="The output result")}
    config_specs = {
        'nb_components': IntParam(default_value=2, min_value=2)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Type[Any]:
        return PCA(n_components=params["nb_components"])

    @classmethod
    def create_result_class(cls) -> Type[PCATrainerResult]:
        return PCATrainerResult
