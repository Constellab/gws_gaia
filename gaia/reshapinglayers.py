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
from tensorflow.keras.layers import Flatten as Kerasflatten
import tensorflow as tf
from gaia.data import Tensor

#==================================================================================
#==================================================================================

class Flatten(Process):
    """
    Flatten layer

    See https://keras.io/api/layers/reshaping_layers/flatten/ for more details
    """
    input_specs = {'tensor' : Tensor}
    output_specs = {'result' : Tensor}
    config_specs = {
    }

    async def task(self):
        x = self.input['tensor']
        y = x._data
        print(np.shape(y))
        z = Kerasflatten()(y)

        t = self.output_specs["result"]
        result = t(tensor=z)
        self.output['result'] = result