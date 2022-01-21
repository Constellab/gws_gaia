# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor as Kerastensor
from tensorflow.keras.layers import Conv1D as Kerasconv1d
from tensorflow.keras.layers import Conv2D as Kerasconv2d
from tensorflow.keras.layers import Conv3D as Kerasconv3d
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam, ListParam)

from gws_core import Dataset
from .data import Tensor

# *****************************************************************************
#
# Conv1D
#
# *****************************************************************************

@task_decorator("TFConv1D", human_name="Convolution 2D",
                short_description="1D-convolution layer (e.g. temporal convolution)")
class Conv1D(Task):
    """
    1D convolution layer (e.g. temporal convolution)

    See https://keras.io/api/layers/convolution_layers/convolution1d/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'nb_filters': IntParam(default_value=32, min_value=0),
        'kernel_size': IntParam(default_value=3, min_value=0),
        'activation_type': StrParam(default_value=None)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = Kerasconv1d(filters=params['nb_filters'], kernel_size=params['kernel_size'], activation=params['activation_type'])(y)
        result = Tensor(result = z)
        return {'result': result}

# *****************************************************************************
#
# Conv2D
#
# *****************************************************************************

@task_decorator("TFConv2D", human_name="Convolution 2D",
                short_description="2D-convolution layer (e.g. spatial convolution over images)")
class Conv2D(Task):
    """
    2D convolution layer (e.g. spatial convolution over images).

    See https://keras.io/api/layers/convolution_layers/convolution2d/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'nb_filters': IntParam(default_value=32, min_value=0),
        'kernel_size': ListParam(default_value=[2, 2]),
        'activation_type': StrParam(default_value=None)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        kernel_size = tuple(params['kernel_size'])
        z = Kerasconv2d(
            filters=params['nb_filters'],
            kernel_size=kernel_size,
            activation=params['activation_type']
        )(y)
        result = Tensor(result = z)
        return {'result': result}

# *****************************************************************************
#
# Conv3D
#
# *****************************************************************************

@task_decorator("TFConv3D", human_name="Convolution 3D",
                short_description="3D-convolution layer (e.g. spatial convolution over volumes)")
class Conv3D(Task):
    """
    3D convolution layer (e.g. spatial convolution over volumes).

    See https://keras.io/api/layers/convolution_layers/convolution3d/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'nb_filters': IntParam(default_value=32, min_value=0),
        'kernel_size': ListParam(default_value=[2, 2, 2]),
        'activation_type': StrParam(default_value=None)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        kernel_size = tuple(params['kernel_size'])
        z = Kerasconv3d(filters=params['nb_filters'], kernel_size=kernel_size, activation=params['activation_type'])(y)
        result = Tensor(result = z)
        return {'result': result}