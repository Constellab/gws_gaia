# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM as KerasLSTM
from tensorflow.keras.layers import GRU as KerasGRU
from tensorflow.keras.layers import SimpleRNN as KerasSimpleRNN
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam, BoolParam)

from .data import Tensor
from ..data.dataset import Dataset

#==================================================================================
#==================================================================================

@task_decorator("LSTM")
class LSTM(Task):
    """
    Long Short-Term Memory (LSTM) layer

    See https://keras.io/api/layers/recurrent_layers/lstm/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'units': IntParam(default_value=10),
        'activation_type': StrParam(default_value='tanh'),
        'recurrent_activation_type': StrParam(default_value='sigmoid'),
        'use_bias': BoolParam(default_value=True)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = KerasLSTM(params['units'], activation=params['activation_type'], recurrent_activation=params['recurrent_activation_type'], use_bias=params['use_bias'])(y)
        result = Tensor(result = z)
        return {'result': result}

#==================================================================================
#==================================================================================

@task_decorator("GRU")
class GRU(Task):
    """
    Gated Recurrent Unit (GRU) layer

    See https://keras.io/api/layers/recurrent_layers/gru/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'units': IntParam(default_value=10),
        'activation_type': StrParam(default_value='tanh'),
        'recurrent_activation_type': StrParam(default_value='sigmoid'),
        'use_bias': BoolParam(default_value=True)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = KerasGRU(params['units'], activation=params['activation_type'], recurrent_activation=params['recurrent_activation_type'], use_bias=params['use_bias'])(y)        
        result = Tensor(result = z)
        return {'result': result}

#==================================================================================
#==================================================================================

@task_decorator("SimpleRNN")
class SimpleRNN(Task):
    """
    Fully-connected RNN where the output is to be fed back to input.

    See https://keras.io/api/layers/recurrent_layers/simple_rnn/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'units': IntParam(default_value=10),
        'activation_type': StrParam(default_value='tanh'),
        'use_bias': BoolParam(default_value=True)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.result
        z = KerasSimpleRNN(params['units'], activation=params['activation_type'], use_bias=params['use_bias'])(y)
        result = Tensor(result = z)
        return {'result': result}