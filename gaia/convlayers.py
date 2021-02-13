# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor as Kerastensor
from tensorflow.keras.layers import Conv1D as Kerasconv1d
from tensorflow.keras.layers import Conv2D as Kerasconv2d
from tensorflow.keras.layers import Conv3D as Kerasconv3d

from gaia.data import Tensor

#==============================================================================
#==============================================================================

class Conv1D(Process):
    """
    1D convolution layer (e.g. temporal convolution).

    See https://keras.io/api/layers/convolution_layers/convolution1d/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'nb_filters': {"type": 'int', "default": 32, "min": 0},
        'kernel_size': {"type": 'int', "default": 3, "min": 0},
        'activation_type': {"type": 'str', "default": None}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasconv1d(filters=self.get_param('nb_filters'), kernel_size=self.get_param('kernel_size'), activation=self.get_param('activation_type'))(y)
        
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#================================================================================
#================================================================================

class Conv2D(Process):
    """
    2D convolution layer (e.g. spatial convolution over images).

    See https://keras.io/api/layers/convolution_layers/convolution2d/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'nb_filters': {"type": 'int', "default": 32, "min": 0},
        'kernel_size': {"type": 'list', "default": [2, 2]},
        'activation_type': {"type": 'str', "default": None}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        kernel_size = tuple(self.get_param('kernel_size'))
        z = Kerasconv2d(filters=self.get_param('nb_filters'), kernel_size=kernel_size, activation=self.get_param('activation_type'))(y)
        
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#================================================================================
#================================================================================

class Conv3D(Process):
    """
    3D convolution layer (e.g. spatial convolution over volumes).
    
    See https://keras.io/api/layers/convolution_layers/convolution3d/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'nb_filters': {"type": 'int', "default": 32, "min": 0},
        'kernel_size': {"type": 'list', "default": [2, 2, 2]},
        'activation_type': {"type": 'str', "default": None}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        kernel_size = tuple(self.get_param('kernel_size'))
        z = Kerasconv3d(filters=self.get_param('nb_filters'), kernel_size=kernel_size, activation=self.get_param('activation_type'))(y)
        
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result