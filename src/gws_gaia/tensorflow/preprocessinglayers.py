# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import tensorflow as tf
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.dataset import Dataset

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

#================================================================================
#================================================================================
