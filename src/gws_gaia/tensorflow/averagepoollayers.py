# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, ListParam,
                      Resource, StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)

from tensorflow.keras.layers import AveragePooling1D as Kerasaveragepooling1d
from tensorflow.keras.layers import AveragePooling2D as Kerasaveragepooling2d
from tensorflow.keras.layers import AveragePooling3D as Kerasaveragepooling3d

from .data import Tensor

# *****************************************************************************
#
# AveragePooling1D
#
# *****************************************************************************


@task_decorator("TFAveragePooling1D", human_name="Average pooling 1D",
                short_description="Average pooling operation for 1D data (e.g. temporal data)")
class AveragePooling1D(Task):
    """
    Average pooling operation for 1D data (temporal data)
    """
    input_specs = {'tensor': InputSpec(Tensor, human_name="Tensor", short_description="The input tensor")}
    output_specs = {'result': OutputSpec(Tensor, human_name="Result", short_description="The output result")}
    config_specs = {
        'pool_size': IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = Kerasaveragepooling1d(pool_size=params['pool_size'])(y)
        result = Tensor(result=z)
        return {'result': result}

# *****************************************************************************
#
# AveragePooling2D
#
# *****************************************************************************


@task_decorator("TFAveragePooling2D", human_name="Average pooling 2D",
                short_description="Average pooling operation for 2D data (e.g. spatial data)")
class AveragePooling2D(Task):
    """
    Average pooling operation for 2D data (spatial data)
    """
    input_specs = {'tensor': InputSpec(Tensor, human_name="Tensor", short_description="The input tensor")}
    output_specs = {'result': OutputSpec(Tensor, human_name="Result", short_description="The output result")}
    config_specs = {
        'pool_size': ListParam(default_value=[2, 2])
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        pool_size = tuple(params['pool_size'])
        z = Kerasaveragepooling2d(pool_size=pool_size)(y)
        result = Tensor(result=z)
        return {'result': result}

# *****************************************************************************
#
# AveragePooling3D
#
# *****************************************************************************


@task_decorator("TFAveragePooling3D", human_name="Average pooling 3D",
                short_description="Average pooling operation for 3D data (e.g. spatial or spatio-temporal data)")
class AveragePooling3D(Task):
    """
    Average pooling operation for 3D data (spatial or spatio-temporal data)
    """
    input_specs = {'tensor': InputSpec(Tensor, human_name="Tensor", short_description="The input tensor")}
    output_specs = {'result': OutputSpec(Tensor, human_name="Result", short_description="The output result")}
    config_specs = {
        'pool_size': ListParam(default_value=[2, 2, 2])
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        pool_size = tuple(params['pool_size'])
        z = Kerasaveragepooling3d(pool_size=pool_size)(y)
        result = Tensor(result=z)
        return {'result': result}
