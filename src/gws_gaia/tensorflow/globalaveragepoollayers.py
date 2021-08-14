# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D as Kerasglobalaveragepooling1d
from tensorflow.keras.layers import GlobalAveragePooling2D as Kerasglobalaveragepooling2d
from tensorflow.keras.layers import GlobalAveragePooling3D as Kerasglobalaveragepooling3d
from pandas import DataFrame

from gws_core import (Process, Resource, ProcessDecorator, ResourceDecorator)

from .data import Tensor
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@ProcessDecorator("GlobalAveragePooling1D")
class GlobalAveragePooling1D(Process):
    """
    Global average pooling operation for 1D data (temporal data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasglobalaveragepooling1d()(y)
        
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#================================================================================
#================================================================================

@ProcessDecorator("GlobalAveragePooling2D")
class GlobalAveragePooling2D(Process):
    """
    Global average pooling operation for 2D data (spatial data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasglobalaveragepooling2d()(y)
        
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result

#================================================================================
#================================================================================

@ProcessDecorator("GlobalAveragePooling3D")
class GlobalAveragePooling3D(Process):
    """
    Global average pooling operation for 3D data (spatial or spatio-temporal data)
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasglobalaveragepooling3d()(y)
 
        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result