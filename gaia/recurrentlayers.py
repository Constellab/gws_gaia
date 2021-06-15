# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.model import Process, Config, Resource

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM as Keraslstm
from tensorflow.keras.layers import GRU as Kerasgru
from tensorflow.keras.layers import SimpleRNN as Kerassimplernn
from gaia.data import Tensor

#==================================================================================
#==================================================================================

class LSTM(Process):
    """
    Long Short-Term Memory (LSTM) layer

    See https://keras.io/api/layers/recurrent_layers/lstm/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'units': {"type": 'int', "default": 10},
        'activation_type': {"type": 'str', "default": 'tanh'},
        'recurrent_activation_type': {"type": 'str', "default": 'sigmoid'},
        'use_bias': {"type": 'bool', "default": True}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Keraslstm(self.get_param('units'), activation=self.get_param('activation_type'), recurrent_activation=self.get_param('recurrent_activation_type'), use_bias=self.get_param('use_bias'))(y)

        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#==================================================================================
#==================================================================================

class GRU(Process):
    """
    Gated Recurrent Unit (GRU) layer

    See https://keras.io/api/layers/recurrent_layers/gru/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'units': {"type": 'int', "default": 10},
        'activation_type': {"type": 'str', "default": 'tanh'},
        'recurrent_activation_type': {"type": 'str', "default": 'sigmoid'},
        'use_bias': {"type": 'bool', "default": True}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasgru(self.get_param('units'), activation=self.get_param('activation_type'), recurrent_activation=self.get_param('recurrent_activation_type'), use_bias=self.get_param('use_bias'))(y)        

        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#==================================================================================
#==================================================================================

class SimpleRNN(Process):
    """
    Fully-connected RNN where the output is to be fed back to input.

    See https://keras.io/api/layers/recurrent_layers/simple_rnn/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'units': {"type": 'int', "default": 10},
        'activation_type': {"type": 'str', "default": 'tanh'},
        'use_bias': {"type": 'bool', "default": True}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerassimplernn(self.get_param('units'), activation=self.get_param('activation_type'), use_bias=self.get_param('use_bias'))(y)

        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result