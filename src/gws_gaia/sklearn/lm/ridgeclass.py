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
from sklearn.linear_model import RidgeClassifier

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# RidgeClassifierResult
#
# *****************************************************************************


@resource_decorator("RidgeClassifierResult", hide=True)
class RidgeClassifierResult(BaseSupervisedResult):
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
# RidgeClassifierTrainer
#
# *****************************************************************************


@task_decorator("RidgeClassifierTrainer", human_name="Ridge classifier trainer",
                short_description="Train a ridge classifier")
class RidgeClassifierTrainer(BaseSupervisedTrainer):
    """
    Trainer of a Ridge regression classifier. Fit a Ridge classifier model with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(RidgeClassifierResult, human_name="result",
                                         short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'alpha': FloatParam(default_value=1, min_value=0),
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return RidgeClassifier(alpha=params["alpha"])

    @classmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        return RidgeClassifierResult

# *****************************************************************************
#
# RidgeClassifierPredictor
#
# *****************************************************************************


@task_decorator("RidgeClassifierPredictor", human_name="Ridge classifier predictor",
                short_description="Predict class labels using a trained Ridge classifier")
class RidgeClassifierPredictor(BaseSupervisedPredictor):
    """
    Predictor of a Ridge regression classifier. Predict class labels for samples in a datatset with a trained Ridge classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        RidgeClassifierResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
