# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (ConfigParams, FloatParam, FloatRField, InputSpec,
                      IntParam, OutputSpec, Resource, StrParam, Table, Task,
                      TaskInputs, TaskOutputs, TechnicalInfo,
                      resource_decorator, task_decorator)
from pandas import DataFrame
from sklearn.linear_model import Ridge

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# RidgeRegressionResult
#
# *****************************************************************************


@resource_decorator("RidgeRegressionResult", hide=True)
class RidgeRegressionResult(BaseSupervisedResult):
    PREDICTION_TABLE_NAME = "Prediction table"
    _r2: int = FloatRField()

    def __init__(self, training_set=None, training_design=None, result=None):
        super().__init__(training_set=training_set, training_design=training_design, result=result)
        if training_set is not None:
            self._create_r2()

    def _create_r2(self) -> float:
        if not self._r2:
            logreg = self.get_result()
            training_set = self.get_training_set()
            training_design = self.get_training_design()
            x_true, y_true = TrainingDesignHelper.create_training_matrices(training_set, training_design)
            self._r2 = logreg.score(
                X=x_true,
                y=y_true
            )
        technical_info = TechnicalInfo(key='R2', value=self._r2)
        self.add_technical_info(technical_info)

# *****************************************************************************
#
# RidgeRegressionTrainer
#
# *****************************************************************************


@task_decorator("RidgeRegressionTrainer", human_name="Ridge regression trainer",
                short_description="Train a ridge regression model")
class RidgeRegressionTrainer(BaseSupervisedTrainer):
    """
    Trainer of a Ridge regression model. Fit a Ridge regression model with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(RidgeRegressionResult, human_name="result",
                                         short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'alpha': FloatParam(default_value=1, min_value=0),
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return Ridge(alpha=params["alpha"])

    @classmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        return RidgeRegressionResult

# *****************************************************************************
#
# RidgeRegressionPredictor
#
# *****************************************************************************


@task_decorator("RidgeRegressionPredictor", human_name="Ridge regression predictor",
                short_description="Predict table targets using a trained ridge regression model")
class RidgeRegressionPredictor(BaseSupervisedPredictor):
    """
    Predictor of a Ridge regression model. Predict target values of a table with a trained Ridge regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        RidgeRegressionResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}