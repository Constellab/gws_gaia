# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.model import Process, Config, Resource

import numpy as np
from tensorflow.keras.layers import GlobalMaxPooling1D as Kerasglobalmaxpooling1d
from tensorflow.keras.layers import GlobalMaxPooling2D as Kerasglobalmaxpooling2d
from tensorflow.keras.layers import GlobalMaxPooling3D as Kerasglobalmaxpooling3d

import tensorflow as tf
from gaia.data import Tensor

#==============================================================================
#==============================================================================

class GlobalMaxPooling1D(Process):
    """
    Global max pooling operation for 1D data (temporal data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasglobalmaxpooling1d()(y)
        
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#================================================================================
#================================================================================

class GlobalMaxPooling2D(Process):
    """
    Global max pooling operation for 2D data (spatial data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasglobalmaxpooling2d()(y)
        
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#================================================================================
#================================================================================

class GlobalMaxPooling3D(Process):
    """
    Global max pooling operation for 3D data (spatial or spatio-temporal data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasglobalmaxpooling3d()(y)
 
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result