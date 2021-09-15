
# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor as KerasTensor
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.models import save_model, load_model

from gws_core import (Task, Resource, task_decorator, resource_decorator, BadRequestException,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam, ListParam)

from dill import dumps, loads

#==============================================================================
#==============================================================================

@resource_decorator("Tensor", hide=True)
class Tensor(Resource):
    def __init__(self, *args, tensor: KerasTensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        # if tensor is not None:
        #     self.binary_store['keras_tensor'] = tensor
        #     #self._data = tensor
        if tensor is not None:
            # dumps(tensor, path)
            self.binary_store['keras_tensor'] = dumps(tensor)

    @property
    def _data(self):
        return loads(self.binary_store['keras_tensor'])
        # return self.binary_store['keras_tensor']

#==============================================================================
#==============================================================================

@resource_decorator("DeepModel", hide=True)
class DeepModel(Resource):
    def __init__(self, *args, model: KerasModel = None,  **kwargs):
        super().__init__(*args, **kwargs)
        
        if not isinstance(model, KerasModel):
            raise BadRequestException(f"The model must an instance of tensorflow.keras.Model. The given model is {model}")
        
        if model is not None:
            path = os.path.join(self.binary_store.full_file_dir, "keras_model")
            save_model(model, path)
            self.binary_store['keras_model_path'] = path
            #self._data = model

    @property
    def _data(self):
        path = self.binary_store['keras_model_path']
        return load_model(path)

#==============================================================================
#==============================================================================

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
        result = Tensor(tensor=y)
        return {'result': result}
