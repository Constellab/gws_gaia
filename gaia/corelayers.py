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
from pandas import DataFrame

from gws.process import Process
from gws.resource import Resource

from .dataset import Dataset
from .data import Tensor

#========================================================================================
#========================================================================================

class Dense(Process):
    """
    Densely connected Neural Network layer.

    See https://keras.io/api/layers/core_layers/dense/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'units': {"type": 'int', "default": 32, "min": 0},
        'activation': {"type": 'str', "default": 'relu'},
        'use_bias': {"type": 'bool', "default": True}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasdense(self.get_param("units"),activation=self.get_param("activation"),use_bias=self.get_param("use_bias"))(y)
        
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#========================================================================================
#========================================================================================

class Activation(Process):
    """
    Applies an activation function to an output.

    See https://keras.io/api/layers/core_layers/activation/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'activation_type': {"type": 'str', "default": 'relu'}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasactivation(self.get_param("activation_type"))(y)
        
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#========================================================================================
#========================================================================================

class Embedding(Process):
    """
    Turns positive integers (indexes) into dense vectors of fixed size.

    See https://keras.io/api/layers/core_layers/embedding/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'input_dimension': {"type": 'int', "default": 1000, "min": 0},
        'output_dimension': {"type": 'int', "default": 64, "min": 0},
        'input_length': {"type": 'int', "default": 10, "min": 0}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasembedding(input_dim=self.get_param("input_dimension"), output_dim=self.get_param("output_dimension"), input_length=self.get_param("input_length"))(y)
        print(z)
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#========================================================================================
#========================================================================================

class Masking(Process):
    """
    Masks a sequence by using a mask value to skip timesteps.
    
    See https://keras.io/api/layers/core_layers/masking/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'mask_value': {"type": 'float', "default": 0.0}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasmasking(mask_value=self.get_param("mask_value"))(y)
        print(z)
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result