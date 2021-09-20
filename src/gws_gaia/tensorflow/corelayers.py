# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense as Kerasdense
from tensorflow.keras.layers import Activation as Kerasactivation
from tensorflow.keras.layers import Embedding as Kerasembedding
from tensorflow.keras.layers import Masking as Kerasmasking
from tensorflow.keras.layers import Dropout as Kerasdropout
from tensorflow.keras.layers import Flatten as Kerasflatten
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam, BoolParam)

from ..data.dataset import Dataset
from .data import Tensor

#========================================================================================
#========================================================================================

@task_decorator("Dense")
class Dense(Task):
    """
    Densely connected Neural Network layer.

    See https://keras.io/api/layers/core_layers/dense/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'units': IntParam(default_value=32, min_value=0),
        'activation': StrParam(default_value='relu'),
        'use_bias': BoolParam(default_value=True)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = Kerasdense(params["units"],activation=params["activation"],use_bias=params["use_bias"])(y)        
        result = Tensor(result = z)
        return {'result': result}

#========================================================================================
#========================================================================================

@task_decorator("Activation")
class Activation(Task):
    """
    Applies an activation function to an output.

    See https://keras.io/api/layers/core_layers/activation/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'activation_type':StrParam(default_value='relu')
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = Kerasactivation(params["activation_type"])(y)
        result = Tensor(result = z)
        return {'result': result}

#========================================================================================
#========================================================================================

@task_decorator("Embedding")
class Embedding(Task):
    """
    Turns positive integers (indexes) into dense vectors of fixed size.

    See https://keras.io/api/layers/core_layers/embedding/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'input_dimension':IntParam(default_value=1000, min_value=0),
        'output_dimension':IntParam(default_value=64, min_value=0),
        'input_length':IntParam(default_value=10, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = Kerasembedding(
            input_dim=params["input_dimension"], 
            output_dim=params["output_dimension"], 
            input_length=params["input_length"])(y)
        result = Tensor(result = z)
        return {'result': result}

#========================================================================================
#========================================================================================

@task_decorator("Masking")
class Masking(Task):
    """
    Masks a sequence by using a mask value to skip timesteps.
    
    See https://keras.io/api/layers/core_layers/masking/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'mask_value': FloatParam(default_value=0.0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = Kerasmasking(mask_value=params["mask_value"])(y)
        result = Tensor(result = z)
        return {'result': result}

@task_decorator("Dropout")
class Dropout(Task):
    """
    Dropout layer

    See https://keras.io/api/layers/regularization_layers/dropout/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
       'rate': FloatParam(default_value=0.5, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = Kerasdropout(params["rate"])(y)
        result = Tensor(result = z)
        return {'result': result}

@task_decorator("Flatten")
class Flatten(Task):
    """
    Flatten layer

    See https://keras.io/api/layers/reshaping_layers/flatten/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = Kerasflatten()(y)
        result = Tensor(result = z)
        return {'result': result}