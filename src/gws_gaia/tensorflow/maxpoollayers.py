# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling1D as Kerasmaxpooling1d
from tensorflow.keras.layers import MaxPooling2D as Kerasmaxpooling2d
from tensorflow.keras.layers import MaxPooling3D as Kerasmaxpooling3d
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator)

from .data import Tensor
from ..data.dataset import Dataset

#==================================================================================
#==================================================================================

@task_decorator("MaxPooling1D")
class MaxPooling1D(Task):
    """
    Max pooling operation for 1D data (temporal data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'pool_size': {"type": 'int', "default": 2, "min": 0}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasmaxpooling1d(pool_size=self.get_param('pool_size'))(y)
 
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#==================================================================================
#==================================================================================

@task_decorator("MaxPooling2D")
class MaxPooling2D(Task):
    """
    Max pooling operation for 2D data (spatial data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'pool_size': {"type": 'list', "default": [2, 2]}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        pool_size = tuple(self.get_param('pool_size'))
        z = Kerasmaxpooling2d(pool_size=pool_size)(y)
        
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#==================================================================================
#==================================================================================

@task_decorator("MaxPooling3D")
class MaxPooling3D(Task):
    """
    Max pooling operation for 3D data (spatial or spatio-temporal data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
        'pool_size': {"type": 'list', "default": [2, 2, 2]}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        pool_size = tuple(self.get_param('pool_size'))
        z = Kerasmaxpooling3d(pool_size=pool_size)(y)
        
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result