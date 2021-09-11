# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import tensorflow as tf
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator)

from ..data.dataset import Dataset

# class Rescaler(Task):
#     input_specs = {'tensor' : Tensor}
#     output_specs = {'result' : Tensor}
#     config_specs = {
#         'pool_size': {"type": 'int', "default": 2, "min": 0}
#     }

#     async def task(self):
#         x = self.input['tensor']
#         y = x._data
#         z = Kerasaveragepooling1d(pool_size=self.get_param('pool_size'))(y)
        
#         t = self.output_specs["result"]
#         result = t(tensor=z)
#         self.output['result'] = result

#================================================================================
#================================================================================
