# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

import numpy as np
from gws_core import (BadRequestException, ConfigParams, DataFrameRField,
                      FloatParam, FloatRField, InputSpec, IntParam, OutputSpec,
                      Resource, ResourceRField, ScatterPlot2DView,
                      ScatterPlot3DView, StrParam, Table, TableView, Task,
                      TaskInputs, TaskOutputs, TechnicalInfo,
                      resource_decorator, task_decorator, view)
from pandas import DataFrame, concat
from sklearn.linear_model import Lasso

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# LassoResult
#
# *****************************************************************************


@resource_decorator("LassoResult", hide=True)
class LassoResult(BaseSupervisedResult):
    """ LassoResult"""

    PREDICTION_TABLE_NAME = "Prediction table"
    _r2: int = FloatRField()

    def __init__(self, training_set=None, training_design=None, result=None):
        super().__init__(training_set=training_set, training_design=training_design, result=result)
        if training_set is not None:
            self._create_r2()

    def _create_r2(self) -> float:
        if not self._r2:
            las = self.get_result()
            training_set = self.get_training_set()
            training_design = self.get_training_design()
            x_true, y_true = TrainingDesignHelper.create_training_matrices(training_set, training_design)
            self._r2 = las.score(
                X=x_true.values,
                y=y_true.values
            )
        technical_info = TechnicalInfo(key='R2', value=self._r2)
        self.add_technical_info(technical_info)

# *****************************************************************************
#
# LassoTrainer
#
# *****************************************************************************


@task_decorator("LassoTrainer", human_name="Lasso trainer",
                short_description="Train a Lasso model")
class LassoTrainer(BaseSupervisedTrainer):
    """
    Trainer of a lasso model. Fit a lasso model with a training table.

    See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.Lasso.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(LassoResult, human_name="result", short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'alpha': FloatParam(default_value=1, min_value=0),
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return Lasso(alpha=params["alpha"])

    @classmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        return LassoResult

# *****************************************************************************
#
# LassoPredictor
#
# *****************************************************************************


@task_decorator("LassoPredictor", human_name="Lasso predictor",
                short_description="Predict table targets using a trained Lasso model")
class LassoPredictor(BaseSupervisedPredictor):
    """
    Predictor of a lasso model. Predict target values from a table with a trained lasso model.

    See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.Lasso.html for more details.
    """
    input_specs = {
        'table': InputSpec(Table, human_name="Table", short_description="The input table"),
        'learned_model': InputSpec(LassoResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
