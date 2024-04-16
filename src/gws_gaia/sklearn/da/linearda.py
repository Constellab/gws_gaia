

from typing import Any, Type

from gws_core import (ConfigParams, InputSpec, IntParam, OutputSpec,
                      ScatterPlot2DView, StrParam, Table, resource_decorator,
                      task_decorator, view, InputSpecs, OutputSpecs)
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedClassResult,
                             BaseSupervisedPredictor, BaseSupervisedTrainer)

# *****************************************************************************
#
# LDATrainerResult
#
# *****************************************************************************


@ resource_decorator("LDATrainerResult",
                     human_name="LDA trainer result",
                     short_description="Linear Discriminant Analysis result",
                     hide=True)
class LDATrainerResult(BaseSupervisedClassResult):
    """ LDATrainerResult """

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
        row_names = self.get_training_set().row_names
        for i, tag in enumerate(row_tags):
            tag["name"] = row_names[i]

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


@ task_decorator("LDATrainer", human_name="LDA trainer",
                 short_description="Train a linear discriminant analysis classifier")
class LDATrainer(BaseSupervisedTrainer):
    """
    Trainer of a linear discriminant analysis classifier. Fit Linear Discriminant Analysis model according to a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(LDATrainerResult, human_name="result", short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'solver': StrParam(default_value='svd'),
        'nb_components': IntParam(default_value=None, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Type[Any]:
        return LinearDiscriminantAnalysis(solver=params["solver"], n_components=params["nb_components"])

    @classmethod
    def create_result_class(cls) -> Type[LDATrainerResult]:
        return LDATrainerResult

# *****************************************************************************
#
# LDAPredictor
#
# *****************************************************************************


@ task_decorator("LDAPredictor", human_name="LDA predictor",
                 short_description="Predict class labels using a Linear Discriminant Analysis (LDA) classifier")
class LDAPredictor(BaseSupervisedPredictor):
    """
    Predictor of a linear discriminant analysis classifier. Predict class labels for samples in a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details.
    """
    input_specs = InputSpecs({
        'table': InputSpec(Table, human_name="Table", short_description="The input table"),
        'learned_model': InputSpec(LDATrainerResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
