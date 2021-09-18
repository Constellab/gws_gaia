# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D as Kerasglobalaveragepooling1d
from tensorflow.keras.layers import GlobalAveragePooling2D as Kerasglobalaveragepooling2d
from tensorflow.keras.layers import GlobalAveragePooling3D as Kerasglobalaveragepooling3d
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from .data import Tensor
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@task_decorator("GlobalAveragePooling1D")
class GlobalAveragePooling1D(Task):
    """
    Global average pooling operation for 1D data (temporal data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = Kerasglobalaveragepooling1d()(y)
        result = Tensor.from_result(result=z)
        return {'result': result}

#================================================================================
#================================================================================

@task_decorator("GlobalAveragePooling2D")
class GlobalAveragePooling2D(Task):
    """
    Global average pooling operation for 2D data (spatial data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = Kerasglobalaveragepooling2d()(y)
        result = Tensor.from_result(result=z)
        return {'result': result}

#================================================================================
#================================================================================

@task_decorator("GlobalAveragePooling3D")
class GlobalAveragePooling3D(Task):
    """
    Global average pooling operation for 3D data (spatial or spatio-temporal data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = Kerasglobalaveragepooling3d()(y)
        result = Tensor.from_result(result=z)
        return {'result': result}