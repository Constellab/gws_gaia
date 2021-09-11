
# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import tensorflow as tf
from tensorflow.python.framework.ops import Tensor as KerasTensor
from tensorflow.keras import Model as KerasModel

from gws_core import (Task, Resource, task_decorator, resource_decorator, BadRequestException)

#==============================================================================
#==============================================================================

@resource_decorator("Tensor", hide=True)
class Tensor(Resource):
    def __init__(self, *args, tensor: KerasTensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        #self.kv_store['tensor'] = tensor
        self._data = tensor

#==============================================================================
#==============================================================================

@resource_decorator("DeepModel", hide=True)
class DeepModel(Resource):
    def __init__(self, *args, model: KerasModel = None,  **kwargs):
        super().__init__(*args, **kwargs)
        
        if not isinstance(model, KerasModel):
            raise BadRequestException(f"The model must an instance of tensorflow.keras.Model. The given model is {model}")

        self._data = model

#==============================================================================
#==============================================================================

@task_decorator("InputConverter")
class InputConverter(Task):
    input_specs = {}
    output_specs = {'result' : Tensor}
    config_specs = {
        'input_shape': {"type": 'list'}
    }

    async def task(self):
        input_shape = tuple(self.get_param('input_shape'))
        y = tf.keras.Input(shape=input_shape)
        
        t = self.output_specs["result"]
        result = t(tensor=y)
        self.output['result'] = result
