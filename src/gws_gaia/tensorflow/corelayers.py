# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (BoolParam, ConfigParams, Dataset, FloatParam, IntParam,
                      Resource, StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)

from tensorflow.keras.layers import Activation as KerasActivation
from tensorflow.keras.layers import Dense as KerasDense
from tensorflow.keras.layers import Dropout as KerasDropout
from tensorflow.keras.layers import Embedding as KerasEmbedding
from tensorflow.keras.layers import Flatten as KerasFlatten
from tensorflow.keras.layers import Masking as KerasMasking

from .data import Tensor

# *****************************************************************************
#
# Dense
#
# *****************************************************************************


@task_decorator("TFDense", human_name="Dense",
                short_description="Densely connected neural network layer")
class Dense(Task):
    """
    Densely connected neural network layer.

    See https://keras.io/api/layers/core_layers/dense/ for more details
    """
    input_specs = {'tensor': InputSpec(Tensor, human_name="Tensor", short_description="The input tensor")}
    output_specs = {'result': OutputSpec(Tensor, human_name="Result", short_description="The output result")}
    config_specs = {
        'units': IntParam(default_value=32, min_value=0),
        'activation': StrParam(default_value='relu'),
        'use_bias': BoolParam(default_value=True)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = KerasDense(params["units"], activation=params["activation"], use_bias=params["use_bias"])(y)
        result = Tensor(result=z)
        return {'result': result}

# *****************************************************************************
#
# Activation
#
# *****************************************************************************


@task_decorator("TFActivation", human_name="Activation",
                short_description="Applies an activation function to an output")
class Activation(Task):
    """
    Applies an activation function to an output.

    See https://keras.io/api/layers/core_layers/activation/ for more details
    """
    input_specs = {'tensor': InputSpec(Tensor, human_name="Tensor", short_description="The input tensor")}
    output_specs = {'result': OutputSpec(Tensor, human_name="Result", short_description="The output result")}
    config_specs = {
        'activation_type': StrParam(default_value='relu')
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = KerasActivation(params["activation_type"])(y)
        result = Tensor(result=z)
        return {'result': result}

# *****************************************************************************
#
# Embedding
#
# *****************************************************************************


@task_decorator("TFEmbedding", human_name="Embedding",
                short_description="Turns positive integers (indexes) into dense vectors of fixed size")
class Embedding(Task):
    """
    Turns positive integers (indexes) into dense vectors of fixed size.

    See https://keras.io/api/layers/core_layers/embedding/ for more details
    """
    input_specs = {'tensor': InputSpec(Tensor, human_name="Tensor", short_description="The input tensor")}
    output_specs = {'result': OutputSpec(Tensor, human_name="Result", short_description="The output result")}
    config_specs = {
        'input_dimension': IntParam(default_value=1000, min_value=0),
        'output_dimension': IntParam(default_value=64, min_value=0),
        'input_length': IntParam(default_value=10, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = KerasEmbedding(
            input_dim=params["input_dimension"],
            output_dim=params["output_dimension"],
            input_length=params["input_length"])(y)
        result = Tensor(result=z)
        return {'result': result}

# *****************************************************************************
#
# Masking
#
# *****************************************************************************


@task_decorator("TFMasking", human_name="Masking",
                short_description="Masks a sequence by using a mask value to skip timesteps")
class Masking(Task):
    """
    Masks a sequence by using a mask value to skip timesteps.

    See https://keras.io/api/layers/core_layers/masking/ for more details
    """
    input_specs = {'tensor': Tensor}
    output_specs = {'result': Tensor}
    config_specs = {
        'mask_value': FloatParam(default_value=0.0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = KerasMasking(mask_value=params["mask_value"])(y)
        result = Tensor(result=z)
        return {'result': result}

# *****************************************************************************
#
# Dropout
#
# *****************************************************************************


@task_decorator("TFDropout", human_name="Dropout",
                short_description="Dropout layer")
class Dropout(Task):
    """
    Dropout layer

    See https://keras.io/api/layers/regularization_layers/dropout/ for more details
    """
    input_specs = {'tensor': Tensor}
    output_specs = {'result': Tensor}
    config_specs = {
        'rate': FloatParam(default_value=0.5, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = KerasDropout(params["rate"])(y)
        result = Tensor(result=z)
        return {'result': result}

# *****************************************************************************
#
# Flatten
#
# *****************************************************************************


@task_decorator("TFFlatten", human_name="Flatten",
                short_description="Flatten layer")
class Flatten(Task):
    """
    Flatten layer

    See https://keras.io/api/layers/reshaping_layers/flatten/ for more details
    """
    input_specs = {'tensor': Tensor}
    output_specs = {'result': Tensor}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = KerasFlatten()(y)
        result = Tensor(result=z)
        return {'result': result}
