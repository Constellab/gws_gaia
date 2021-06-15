# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.model import Process, Config, Resource

import numpy as np
from tensorflow.keras.layers import Dropout as Kerasdropout
import tensorflow as tf
from gaia.data import Tensor

#==================================================================================
#==================================================================================

class Dropout(Process):
    """
    Dropout layer

    See https://keras.io/api/layers/regularization_layers/dropout/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
       'rate': {"type": 'float', "default": 0.5, "min": 0}
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        z = Kerasdropout(self.get_param("rate"))(y)

        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result