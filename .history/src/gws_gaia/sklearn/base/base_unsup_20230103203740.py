# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (ConfigParams, IntParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator)

from ...base.base_resource import BaseResourceSet

# *****************************************************************************
#
# AdaBoostClassifierResult
#
# *****************************************************************************


@resource_decorator("BaseUnsupervisedResult", hide=True)
class BaseUnsupervisedResult(BaseResourceSet):
    """BaseUnsupervisedResult"""
        
# *****************************************************************************
#
# AdaBoostClassifierTrainer
#
# *****************************************************************************


@task_decorator("BaseUnsupervisedTrainer", human_name="Base unsupervised trainer", hide=True)
class BaseUnsupervisedTrainer(Task):
    """BaseUnsupervisedTrainer"""

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return None

    @classmethod
    def create_result_class(cls) -> Type[BaseUnsupervisedResult]:
        return None

    @classmethod
    def fit(cls, table: Table, training_design: Dict) -> BaseSupervisedResult:
        sklearn_trainer = self.create_sklearn_trainer_class(params)
        x_true, y_true = TrainingDesignHelper.create_training_matrices(table, training_design, dummy=self._dummy_target)
        if y_true is None:
            self.log_error_message("No target Y defined in the training design")
        sklearn_trainer.fit(x_true, y_true)
        return sklearn_trainer

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        sklearn_trainer = self.create_sklearn_trainer_class(params)
        
        return {"result": result}

# *****************************************************************************
#
# BaseSupervisedPredictor
#
# *****************************************************************************


@ task_decorator("BaseUnsupervisedPredictor", human_name="Base unsupervised predictor", hide=True)
class BaseUnsupervisedPredictor(Task):
    """BaseUnsupervisedPredictor"""