
# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from typing import Any

from gws_core import (BadRequestException, ConfigParams, FloatParam, IntParam,
                      ListParam, Resource, RField, StrParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, InputSpec)

import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.python.framework.ops import Tensor as KerasTensor

# *****************************************************************************
#
# DeepResult
#
# *****************************************************************************


@resource_decorator("DeepResult", hide=True)
class DeepResult(Resource):
    _result: Any = RField(default_value=None)

    def __init__(self, result=None):
        super().__init__()
        if result is not None:
            self._result = result

    def get_result(self):
        return self._result

# *****************************************************************************
#
# Tensor
#
# *****************************************************************************


@resource_decorator("Tensor", hide=True)
class Tensor(DeepResult):
    pass

# *****************************************************************************
#
# DeepModel
#
# *****************************************************************************


@resource_decorator("DeepModel", hide=True)
class DeepModel(Resource):
    pass

# *****************************************************************************
#
# InputConverter
#
# *****************************************************************************

@task_decorator("InputConverter", human_name="Input converter",
                short_description="Input converter")
class InputConverter(Task):
    input_specs = {}
    output_specs = {'result': InputSpec(Tensor, human_name="Tensor", short_description="The output tensor")}
    config_specs = {
        'input_shape': ListParam()
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        input_shape = tuple(params['input_shape'])
        y = tf.keras.Input(shape=input_shape)
        result = Tensor(result=y)
        return {'result': result}
