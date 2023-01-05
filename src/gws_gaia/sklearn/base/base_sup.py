# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from abc import abstractmethod
from typing import Any, Dict, Type

from gws_core import (ConfigParams, FloatRField, Table, Task, TaskInputs,
                      TaskOutputs, TechnicalInfo, resource_decorator,
                      task_decorator)
from pandas import DataFrame
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from ...base.base_resource import BaseResourceSet
from ...base.helper.training_design_helper import TrainingDesignHelper

# *****************************************************************************
#
# AdaBoostClassifierResult
#
# *****************************************************************************


@resource_decorator("BaseSupervisedResult", hide=True)
class BaseSupervisedResult(BaseResourceSet):
    """BaseSupervisedResult"""

    PREDICTION_TABLE_NAME = "Prediction table"
    PREDICTION_SCORE_NAME = "Prediction score"

    _dummy_target = False
    _prediction_score: int = FloatRField()

    def __init__(self, training_set=None, training_design=None, result=None):
        super().__init__(training_set=training_set, training_design=training_design, result=result)
        if training_set is not None:
            self._create_prediction_table()
            self._create_prediction_score()

    def _create_prediction_table(self) -> DataFrame:
        mdl = self.get_result()
        training_set = self.get_training_set()
        training_design = self.get_training_design()
        table = TrainingDesignHelper.predict(mdl, training_set, training_design, dummy=self._dummy_target)
        table.name = self.PREDICTION_TABLE_NAME
        self.add_resource(table)

    def _create_prediction_score(self) -> float:
        if not self._prediction_score:
            sklearn_mdl = self.get_result()
            training_set = self.get_training_set()
            training_design = self.get_training_design()
            x_true, y_true = TrainingDesignHelper.create_training_matrices(
                training_set, training_design, dummy=self._dummy_target)
            self._prediction_score = sklearn_mdl.score(
                X=x_true,
                y=y_true
            )
        technical_info = TechnicalInfo(key=self.PREDICTION_SCORE_NAME, value=self._prediction_score)
        self.add_technical_info(technical_info)

    def get_prediction_table(self):
        """ Get prediction table """
        if self.resource_exists(self.PREDICTION_TABLE_NAME):
            return self.get_resource(self.PREDICTION_TABLE_NAME)
        else:
            return None

    def get_prediction_score(self):
        return self._prediction_score

    def predict(self, table: Table) -> Table:
        mdl = self.get_result()
        y_pred = mdl.predict(table.get_data())
        training_set = self.get_training_set()
        training_design = self.get_training_design()
        _, y_true = TrainingDesignHelper.create_training_matrices(
            training_set, training_design, dummy=self._dummy_target)
        pred_table = Table(
            data=y_pred,
            row_names=table.row_names,
            column_names=list(y_true.columns)
        )
        return pred_table


@resource_decorator("BaseSupervisedRegResult", hide=True)
class BaseSupervisedRegResult(BaseSupervisedResult):
    """BaseSupervisedResult"""

    PREDICTION_SCORE_NAME = "R2"


@resource_decorator("BaseSupervisedClassResult", hide=True)
class BaseSupervisedClassResult(BaseSupervisedResult):
    """BaseSupervisedResult"""

    PREDICTION_SCORE_NAME = "Mean accuracy"

# *****************************************************************************
#
# AdaBoostClassifierTrainer
#
# *****************************************************************************


@task_decorator("BaseSupervisedTrainer", human_name="Base supervised trainer", hide=True)
class BaseSupervisedTrainer(Task):
    """BaseSupervisedTrainer"""

    _dummy_target = False

    @classmethod
    @abstractmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        pass

    @classmethod
    @abstractmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        pass

    def fit(self, table: Table, params: Dict) -> BaseSupervisedResult:
        training_design = params["training_design"]
        sklearn_trainer = self.create_sklearn_trainer_class(params)
        x_true, y_true = TrainingDesignHelper.create_training_matrices(table, training_design, dummy=self._dummy_target)
        if y_true is None:
            self.log_error_message("No target Y defined in the training design")
        sklearn_trainer.fit(x_true, y_true)
        return sklearn_trainer

    def fit_cv(self, table: Table, params: Dict) -> BaseSupervisedResult:
        pass

    def fit_cv_search(self, table: Table, params: Dict) -> BaseSupervisedResult:
        pass

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs["table"]
        training_design = params["training_design"]
        sklearn_trainer = self.fit(table, params)
        result_class: BaseSupervisedResult = self.create_result_class()
        result = result_class(training_set=table, training_design=training_design, result=sklearn_trainer)
        return {"result": result}

# *****************************************************************************
#
# BaseSupervisedPredictor
#
# *****************************************************************************


@ task_decorator("BaseSupervisedPredictor", human_name="Base supervised predictor", hide=True)
class BaseSupervisedPredictor(Task):
    """BaseSupervisedPredictor"""

    _dummy_target = False

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        learned_model = inputs['learned_model']
        pred_table = learned_model.predict(table)
        return {'result': pred_table}
