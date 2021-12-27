
# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
from typing import Any
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor as KerasTensor
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.models import save_model, load_model

from gws_core import (Task, Resource, task_decorator, resource_decorator, 
                        BadRequestException, ConfigParams, TaskInputs, 
                        TaskOutputs, IntParam, FloatParam, StrParam, ListParam, RField)
from ..data.core import GenericResult

# *****************************************************************************
#
# Tensor
#
# *****************************************************************************

@resource_decorator("Tensor", hide=True)
class Tensor(GenericResult):
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

@task_decorator("InputConverter")
class InputConverter(Task):
    input_specs = {}
    output_specs = {'result' : Tensor}
    config_specs = {
        'input_shape': ListParam()
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        input_shape = tuple(params['input_shape'])
        y = tf.keras.Input(shape=input_shape)
        result = Tensor(result = y)
        return {'result': result}
