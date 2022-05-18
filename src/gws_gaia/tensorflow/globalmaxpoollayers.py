# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, ListParam,
                      Resource, StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)

from tensorflow.keras.layers import \
    GlobalMaxPooling1D as Kerasglobalmaxpooling1d
from tensorflow.keras.layers import \
    GlobalMaxPooling2D as Kerasglobalmaxpooling2d
from tensorflow.keras.layers import \
    GlobalMaxPooling3D as Kerasglobalmaxpooling3d

from .data import Tensor

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
    input_specs = {'tensor': InputSpec(Tensor, human_name="Tensor", short_description="The input tensor")}
    output_specs = {'result': OutputSpec(Tensor, human_name="Result", short_description="The output result")}
    config_specs = {'pool_size': IntParam(default_value=2, min_value=0)}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = Kerasglobalmaxpooling1d()(y)
        result = Tensor(result=z)
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
    input_specs = {'tensor': InputSpec(Tensor, human_name="Tensor", short_description="The input tensor")}
    output_specs = {'result': OutputSpec(Tensor, human_name="Result", short_description="The output result")}
    config_specs = {'pool_size': ListParam(default_value=[2, 2])}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = Kerasglobalmaxpooling2d()(y)
        result = Tensor(result=z)
        return {'result': result}

# *****************************************************************************
#
# GlobalMaxPooling3D
#
# *****************************************************************************


@task_decorator("GlobalMaxPooling3D", human_name="Global max pooling 3D",
                short_description="Global max pooling operation for 3D data (spatial or spatio-temporal data)")
class GlobalMaxPooling3D(Task):
    """
    Global max pooling operation for 3D data (spatial or spatio-temporal data)
    """
    input_specs = {'tensor': InputSpec(Tensor, human_name="Tensor", short_description="The input tensor")}
    output_specs = {'result': OutputSpec(Tensor, human_name="Result", short_description="The output result")}
    config_specs = {'pool_size': ListParam(default_value=[2, 2, 2])}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['tensor']
        y = x.get_result()
        z = Kerasglobalmaxpooling3d()(y)
        result = Tensor(result=z)
        return {'result': result}
