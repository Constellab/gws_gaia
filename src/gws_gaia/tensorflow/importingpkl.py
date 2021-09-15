
# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import sys, pickle
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor as Kerastensor
from tensorflow.keras import Model as KerasModel

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import Tuple

#==============================================================================
#==============================================================================

@task_decorator("ImporterPKL")
class ImporterPKL(Task):
    input_specs = {}
    output_specs = {'result': Tuple}
    config_specs = {
        'file_path': StrParam(default_value=""),
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        f = open(params['file_path'], 'rb')
        if sys.version_info < (3,):
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='bytes')
        f.close()
        result = Tuple(tup=data)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("Preprocessor")
class Preprocessor(Task):
    input_specs = {'data': Tuple}
    output_specs = {'result': Tuple}
    config_specs = {
        'number_classes':IntParam(default_value=10, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['data']
        data = x._data

        (x_train, y_train), (x_test, y_test) = data  
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        y_train = tf.keras.utils.to_categorical(y_train, params['number_classes'])
        y_test = tf.keras.utils.to_categorical(y_test, params['number_classes'])
        data = (x_train, y_train), (x_test, y_test)
        result = Tuple(tup=data)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("AdhocExtractor")
class AdhocExtractor(Task):
    input_specs = {'data': Tuple}
    output_specs = {'result': Tuple}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['data']
        data = x._data
        (x_train, _), (_, _) = data  
        data = x_train
        result = Tuple(tup=data)
        return {'result': result}
