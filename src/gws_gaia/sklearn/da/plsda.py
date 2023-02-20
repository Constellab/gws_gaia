# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, List, Type

from gws_core import (ConfigParams, InputSpec, IntParam, OutputSpec,
                      ScatterPlot2DView, Table, TechnicalInfo,
                      resource_decorator, task_decorator, view)
from pandas import DataFrame
from sklearn.cross_decomposition import PLSRegression

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedClassResult,
                             BaseSupervisedPredictor, BaseSupervisedTrainer)
from ..decomp.helper.pls_helper import PLSHelper

# *****************************************************************************
#
# PLSDATrainerResult
#
# *****************************************************************************


@resource_decorator("PLSDATrainerResult", hide=True)
class PLSDATrainerResult(BaseSupervisedClassResult):
    """ PLSDATrainerResult """
    TRANSFORMED_TABLE_NAME = "Transformed table"
    VARIANCE_TABLE_NAME = "Variance table"
    PREDICTION_TABLE_NAME = "Prediction table"
    _dummy_target = True

    def __init__(self, training_set=None, training_design=None, result=None):
        super().__init__(training_set=training_set, training_design=training_design, result=result)
        if training_set is not None:
            self._create_transformed_table()
            self._create_variance_table()

    def _create_transformed_table(self):
        pls: PLSRegression = self.get_result()
        ncomp = pls.x_rotations_.shape[1]

        training_set = self.get_training_set()
        training_design = self.get_training_design()
        x_true, _ = TrainingDesignHelper.create_training_matrices(training_set, training_design)

        data: DataFrame = pls.transform(x_true.values)
        columns = [f"PC{i+1}" for i in range(0, ncomp)]
        data = DataFrame(data=data, columns=columns, index=self.get_training_set().row_names)
        table = Table(data=data)
        row_tags = self.get_training_set().get_row_tags()
        table.name = self.TRANSFORMED_TABLE_NAME
        table.set_all_row_tags(row_tags)
        self.add_resource(table)

    def _create_variance_table(self) -> List[float]:
        pls: PLSRegression = self.get_result()
        training_set = self.get_training_set()
        training_design = self.get_training_design()
        table = PLSHelper.create_variance_table(pls, training_set, training_design, dummy=self._dummy_target)

        table.name = self.VARIANCE_TABLE_NAME
        self.add_resource(table)
        data = table.get_data()
        self.add_technical_info(TechnicalInfo(key='PC1', value=f'{data.iat[0,0]:.3f}'))
        self.add_technical_info(TechnicalInfo(key='PC2', value=f'{data.iat[1,0]:.3f}'))

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

        data: DataFrame = self.get_transformed_table().get_data()
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
        var = self.get_variance_table().get_data()
        _view.x_label = f'PC1 ({100*var.iat[0,0]:.2f}%)'
        _view.y_label = f'PC2 ({100*var.iat[1,0]:.2f}%)'
        return _view

# *****************************************************************************
#
# PLSDATrainer
#
# *****************************************************************************


@ task_decorator("PLSDATrainer", human_name="PLSDA trainer",
                 short_description="Train a Partial Least Squares Discrimante Analysis (PLSDA) regression model")
class PLSDATrainer(BaseSupervisedTrainer):
    """
    Trainer of a Partial Least Squares Discrimante Analysis (PLSDA) regression model. Fit a PLSDA regression model to a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details.
    """

    _dummy_target = True

    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(PLSDATrainerResult, human_name="result",
                                         short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'nb_components': IntParam(default_value=2, min_value=0),
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Type[Any]:
        return PLSRegression(n_components=params["nb_components"])

    @classmethod
    def create_result_class(cls) -> Type[PLSDATrainerResult]:
        return PLSDATrainerResult

# *****************************************************************************
#
# PLSDAPredictor
#
# *****************************************************************************


@ task_decorator("PLSDAPredictor", human_name="PLSDA predictor",
                 short_description="Predict table targets using a trained PLSDA regression model")
class PLSDAPredictor(BaseSupervisedPredictor):
    """
    Predictor of a Partial Least Squares Discrimante Analysis (PLSDA) regression model. Predict targets of a table with a trained PLSDA regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details.
    """

    _dummy_target = True

    input_specs = {
        'table': InputSpec(Table, human_name="Table", short_description="The input table"),
        'learned_model': InputSpec(PLSDATrainerResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
