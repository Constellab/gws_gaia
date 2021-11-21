# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling1D as Kerasmaxpooling1d
from tensorflow.keras.layers import MaxPooling2D as Kerasmaxpooling2d
from tensorflow.keras.layers import MaxPooling3D as Kerasmaxpooling3d
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam, ListParam)

from .data import Tensor
from gws_core import Dataset

#==================================================================================
#==================================================================================

@task_decorator("MaxPooling1D")
class MaxPooling1D(Task):
    """
    Max pooling operation for 1D data (temporal data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'pool_size':IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = Kerasmaxpooling1d(pool_size=params['pool_size'])(y)
        result = Tensor(result = z)
        return {'result': result}

#==================================================================================
#==================================================================================

@task_decorator("MaxPooling2D")
class MaxPooling2D(Task):
    """
    Max pooling operation for 2D data (spatial data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'pool_size': ListParam(default_value=[2, 2])
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        pool_size = tuple(params['pool_size'])
        z = Kerasmaxpooling2d(pool_size=pool_size)(y)
        result = Tensor(result = z)
        return {'result': result}

#==================================================================================
#==================================================================================

@task_decorator("MaxPooling3D")
class MaxPooling3D(Task):
    """
    Max pooling operation for 3D data (spatial or spatio-temporal data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'pool_size': ListParam(default_value=[2, 2, 2])
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        pool_size = tuple(params['pool_size'])
        z = Kerasmaxpooling3d(pool_size=pool_size)(y)
        result = Tensor(result = z)
        return {'result': result}