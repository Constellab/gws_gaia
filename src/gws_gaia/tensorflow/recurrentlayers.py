# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (BoolParam, ConfigParams, Dataset, FloatParam, IntParam,
                      Resource, StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)
from pandas import DataFrame

import tensorflow as tf
from tensorflow.keras.layers import GRU as KerasGRU
from tensorflow.keras.layers import LSTM as KerasLSTM
from tensorflow.keras.layers import SimpleRNN as KerasSimpleRNN

from .data import Tensor

# *****************************************************************************
#
# LSTM
#
# *****************************************************************************


@task_decorator("LSTM", human_name="LSTM",
                short_description="Long Short-Term Memory (LSTM) layer")
class LSTM(Task):
    """
    Long Short-Term Memory (LSTM) layer

    See https://keras.io/api/layers/recurrent_layers/lstm/ for more details
    """
    input_specs = {'tensor': InputSpec(Tensor, human_name="Tensor", short_description="The input tensor")}
    output_specs = {'result': OutputSpec(Tensor, human_name="Result", short_description="The output result")}
    config_specs = {
        'units': IntParam(default_value=10),
        'activation_type': StrParam(default_value='tanh'),
        'recurrent_activation_type': StrParam(default_value='sigmoid'),
        'use_bias': BoolParam(default_value=True)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = KerasLSTM(
            params['units'],
            activation=params['activation_type'],
            recurrent_activation=params['recurrent_activation_type'],
            use_bias=params['use_bias'])(y)
        result = Tensor(result=z)
        return {'result': result}

# *****************************************************************************
#
# GRU
#
# *****************************************************************************


@task_decorator("GRU", human_name="GRU",
                short_description="Gated Recurrent Unit (GRU) layer")
class GRU(Task):
    """
    Gated Recurrent Unit (GRU) layer

    See https://keras.io/api/layers/recurrent_layers/gru/ for more details
    """
    input_specs = {'tensor': InputSpec(Tensor, human_name="Tensor", short_description="The input tensor")}
    output_specs = {'result': OutputSpec(Tensor, human_name="Result", short_description="The output result")}
    config_specs = {
        'units': IntParam(default_value=10),
        'activation_type': StrParam(default_value='tanh'),
        'recurrent_activation_type': StrParam(default_value='sigmoid'),
        'use_bias': BoolParam(default_value=True)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = KerasGRU(
            params['units'],
            activation=params['activation_type'],
            recurrent_activation=params['recurrent_activation_type'],
            use_bias=params['use_bias'])(y)
        result = Tensor(result=z)
        return {'result': result}

# *****************************************************************************
#
# SimpleRNN
#
# *****************************************************************************


@task_decorator("SimpleRNN", human_name="SimpleRNN",
                short_description="Fully-connected RNN where the output is to be fed back to input")
class SimpleRNN(Task):
    """
    Fully-connected RNN where the output is to be fed back to input.

    See https://keras.io/api/layers/recurrent_layers/simple_rnn/ for more details
    """
    input_specs = {'tensor': InputSpec(Tensor, human_name="Tensor", short_description="The input tensor")}
    output_specs = {'result': OutputSpec(Tensor, human_name="Result", short_description="The output result")}
    config_specs = {
        'units': IntParam(default_value=10),
        'activation_type': StrParam(default_value='tanh'),
        'use_bias': BoolParam(default_value=True)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = KerasSimpleRNN(params['units'], activation=params['activation_type'], use_bias=params['use_bias'])(y)
        result = Tensor(result=z)
        return {'result': result}
