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

    def __init__(self, training_set=None, training_design=None, result=None):
        super().__init__(training_set=training_set, training_design=training_design, result=result)
        if training_set is not None:
            self._create_prediction_table()
            
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

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        sklearn_trainer = self.create_sklearn_trainer_class(params)
        sklearn_trainer.fit(table.get_data())
        result_class = self.create_result_class()
        result = result_class(training_set=table, result=sklearn_trainer)
        return {"result": result}

# *****************************************************************************
#
# BaseSupervisedPredictor
#
# *****************************************************************************


@ task_decorator("BaseUnsupervisedPredictor", human_name="Base unsupervised predictor", hide=True)
class BaseUnsupervisedPredictor(Task):
    """BaseUnsupervisedPredictor"""
