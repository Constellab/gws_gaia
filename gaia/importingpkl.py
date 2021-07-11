
# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import sys, pickle
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor as Kerastensor
from tensorflow.keras import Model as KerasModel

from gws.process import Process
from gws.resource import Resource

from .data import Tuple

#==============================================================================
#==============================================================================

class ImporterPKL(Process):
    input_specs = {}
    output_specs = {'result': Tuple}
    config_specs = {
        'file_path': {"type": 'str', "default": ""},
    }

    async def task(self):
        f = open(self.get_param('file_path'), 'rb')
        if sys.version_info < (3,):
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='bytes')
        f.close()

        t = self.output_specs["result"]
        result = t(tup=data)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class Preprocessor(Process):
    input_specs = {'data': Tuple}
    output_specs = {'result': Tuple}
    config_specs = {
        'number_classes': {"type": 'int', "default": 10, "min": 0}
    }

    async def task(self):
        x = self.input['data']
        data = x._data
        (x_train, y_train), (x_test, y_test) = data  
        print(y_train[0:4])

        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        y_train = tf.keras.utils.to_categorical(y_train, self.get_param('number_classes'))
        y_test = tf.keras.utils.to_categorical(y_test, self.get_param('number_classes'))
        
        data = (x_train, y_train), (x_test, y_test)
        t = self.output_specs["result"]
        result = t(tup=data)
        self.output['result'] = result

#==============================================================================
#==============================================================================

class AdhocExtractor(Process):
    input_specs = {'data': Tuple}
    output_specs = {'result': Tuple}
    config_specs = {
    }

    async def task(self):
        x = self.input['data']
        data = x._data
        (x_train, _), (_, _) = data  
        
        data = x_train
        t = self.output_specs["result"]
        result = t(tup=data)
        self.output['result'] = result
