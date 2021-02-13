# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling1D as Kerasmaxpooling1d
from tensorflow.keras.layers import MaxPooling2D as Kerasmaxpooling2d
from tensorflow.keras.layers import MaxPooling3D as Kerasmaxpooling3d
from gaia.data import Tensor

#==================================================================================
#==================================================================================

class MaxPooling1D(Process):
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

class MaxPooling2D(Process):
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

class MaxPooling3D(Process):
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