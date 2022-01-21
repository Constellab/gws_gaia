# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import GlobalMaxPooling1D as Kerasglobalmaxpooling1d
from tensorflow.keras.layers import GlobalMaxPooling2D as Kerasglobalmaxpooling2d
from tensorflow.keras.layers import GlobalMaxPooling3D as Kerasglobalmaxpooling3d
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam, ListParam)

from .data import Tensor
from gws_core import Dataset

# *****************************************************************************
#
# GlobalMaxPooling1D
#
# *****************************************************************************

@task_decorator("GlobalMaxPooling1D", human_name="Global max pooling 1D",
                short_description="Global max pooling operation for 1D data (temporal data)")
class GlobalMaxPooling1D(Task):
    """
    Global max pooling operation for 1D data (temporal data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {'pool_size': IntParam(default_value=2, min_value=0)}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = Kerasglobalmaxpooling1d()(y)
        result = Tensor(result = z)
        return {'result': result}

# *****************************************************************************
#
# GlobalMaxPooling2D
#
# *****************************************************************************

@task_decorator("GlobalMaxPooling2D", human_name="Global max pooling 2D",
                short_description="Global max pooling operation for 2D data (spatial data)")
class GlobalMaxPooling2D(Task):
    """
    Global max pooling operation for 2D data (spatial data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {'pool_size': ListParam(default_value=[2,2])}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = Kerasglobalmaxpooling2d()(y)
        result = Tensor(result = z)
        return {'result': result}

# *****************************************************************************
#
# GlobalMaxPooling2D
#
# *****************************************************************************

@task_decorator("GlobalMaxPooling3D", human_name="Global max pooling 3D",
                short_description="Global max pooling operation for 3D data (spatial or spatio-temporal data)")
class GlobalMaxPooling3D(Task):
    """
    Global max pooling operation for 3D data (spatial or spatio-temporal data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {'pool_size': ListParam(default_value=[2,2,2])}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = Kerasglobalmaxpooling3d()(y)
        result = Tensor(result = z)
        return {'result': result}