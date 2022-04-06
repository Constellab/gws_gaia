# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator)
from pandas import DataFrame

import tensorflow as tf

# class Rescaler(Task):
#     input_specs = {'tensor' : Tensor}
#     output_specs = {'result' : Tensor}
#     config_specs = {
#         'pool_size':IntParam(default_value=2, min_value=0}
#     }

#     async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
#         x = inputs['tensor']
#         y = x.result
#         z = Kerasaveragepooling1d(pool_size=params['pool_size'])(y)

#
#         result = t(tensor=z)
#         return {'result': result}

# ================================================================================
# ================================================================================
