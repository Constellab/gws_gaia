# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, ListParam,
                      Resource, StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator)

from tensorflow.keras.layers import MaxPooling1D as Kerasmaxpooling1d
from tensorflow.keras.layers import MaxPooling2D as Kerasmaxpooling2d
from tensorflow.keras.layers import MaxPooling3D as Kerasmaxpooling3d

from .data import Tensor

# *****************************************************************************
#
# MaxPooling1D
#
# *****************************************************************************


@task_decorator("MaxPooling1D", human_name="Max pooling 1D",
                short_description="Max pooling operation for 1D data (temporal data)")
class MaxPooling1D(Task):
    """
    Max pooling operation for 1D data (temporal data)
    """
    input_specs = {'tensor': Tensor}
    output_specs = {'result': Tensor}
    config_specs = {
        'pool_size': IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = Kerasmaxpooling1d(pool_size=params['pool_size'])(y)
        result = Tensor(result=z)
        return {'result': result}

# *****************************************************************************
#
# MaxPooling2D
#
# *****************************************************************************


@task_decorator("MaxPooling2D", human_name="Max pooling 2D",
                short_description="Max pooling operation for 2D data (spatial data)")
class MaxPooling2D(Task):
    """
    Max pooling operation for 2D data (spatial data)
    """
    input_specs = {'tensor': Tensor}
    output_specs = {'result': Tensor}
    config_specs = {
        'pool_size': ListParam(default_value=[2, 2])
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        pool_size = tuple(params['pool_size'])
        z = Kerasmaxpooling2d(pool_size=pool_size)(y)
        result = Tensor(result=z)
        return {'result': result}

# *****************************************************************************
#
# MaxPooling3D
#
# *****************************************************************************


@task_decorator("MaxPooling3D", human_name="Max pooling 3D",
                short_description="Max pooling operation for 3D data (spatial or spatio-temporal data)")
class MaxPooling3D(Task):
    """
    Max pooling operation for 3D data (spatial or spatio-temporal data)
    """
    input_specs = {'tensor': Tensor}
    output_specs = {'result': Tensor}
    config_specs = {
        'pool_size': ListParam(default_value=[2, 2, 2])
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        pool_size = tuple(params['pool_size'])
        z = Kerasmaxpooling3d(pool_size=pool_size)(y)
        result = Tensor(result=z)
        return {'result': result}
