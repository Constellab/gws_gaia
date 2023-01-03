# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type, List

from gws_core import (ConfigParams, Table, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator)
from pandas import DataFrame

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
    _dummy_target = False

    def __init__(self, training_set=None, training_design=None, result=None):
        super().__init__(training_set=training_set, training_design=training_design, result=result)
        if training_set is not None:
            self._create_prediction_table()

    def _create_prediction_table(self) -> DataFrame:
        mdl = self.get_result()
        training_set = self.get_training_set()
        training_design = self.get_training_design()
        table = TrainingDesignHelper.predict(mdl, training_set, training_design, dummy=self._dummy_target)
        table.name = self.PREDICTION_TABLE_NAME
        self.add_resource(table)

    def get_prediction_table(self):
        """ Get prediction table """
        if self.resource_exists(self.PREDICTION_TABLE_NAME):
            return self.get_resource(self.PREDICTION_TABLE_NAME)
        else:
            return None

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
    def create_sklearn_trainer_class(cls, params) -> Any:
        return None

    @classmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        return None

    @classmethod
    def fit(cls, params: ConfigParams, inputs: TaskInputs) -> BaseSupervisedResult:
        table = inputs['table']
        training_design = params["training_design"]
        sklearn_trainer = self.create_sklearn_trainer_class(params)
        x_true, y_true = TrainingDesignHelper.create_training_matrices(table, training_design, dummy=self._dummy_target)
        if y_true is None:
            self.log_error_message("No target Y defined in the training design")
        sklearn_trainer.fit(x_true, y_true)
        return sklearn_trainer

    @classmethod
    def predict(cls, learned_model: BaseSupervisedResult, table: Table) -> Table:
        mdl = learned_model.get_result()
        y_pred = mdl.predict(table.get_data())
        training_set = learned_model.get_training_set()
        training_design = learned_model.get_training_design()
        _, y_true = TrainingDesignHelper.create_training_matrices(
            training_set, training_design, dummy=self._dummy_target)
        result_table = Table(
            data=y_pred,
            row_names=table.row_names,
            column_names=list(y_true.columns)
        )
        return result_table

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        sklearn_trainer = self.fit(params, training_design)
        result_class = self.create_result_class()
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
        result_table = self.predict(learned_model, table)
        return {'result': result_table}
