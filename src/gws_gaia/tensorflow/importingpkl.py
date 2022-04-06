
# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import pickle
import sys

from gws_core import (ConfigParams, FloatParam, IntParam, Resource, StrParam,
                      Task, TaskInputs, TaskOutputs, resource_decorator,
                      task_decorator)

import tensorflow as tf
from tensorflow.python.framework.ops import Tensor as Kerastensor

from .data import DeepResult

# *****************************************************************************
#
# PickleImporter
#
# *****************************************************************************


@task_decorator("PickleImporter", human_name="Pickle importer")
class PickleImporter(Task):
    input_specs = {}
    output_specs = {'result': DeepResult}
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
        result = DeepResult(result=data)
        return {'result': result}

# *****************************************************************************
#
# Preprocessor
#
# *****************************************************************************


@task_decorator("Preprocessor")
class Preprocessor(Task):
    input_specs = {'data': DeepResult}
    output_specs = {'result': DeepResult}
    config_specs = {
        'number_classes': IntParam(default_value=10, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['data']
        data = x.get_result()

        (x_train, y_train), (x_test, y_test) = data
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        y_train = tf.keras.utils.to_categorical(y_train, params['number_classes'])
        y_test = tf.keras.utils.to_categorical(y_test, params['number_classes'])
        data = (x_train, y_train), (x_test, y_test)
        result = DeepResult(result=data)
        return {'result': result}

# *****************************************************************************
#
# AdhocExtractor
#
# *****************************************************************************


@task_decorator("AdhocExtractor")
class AdhocExtractor(Task):
    input_specs = {'data': DeepResult}
    output_specs = {'result': DeepResult}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['data']
        data = x.get_result()
        (x_train, _), (_, _) = data
        data = x_train
        result = DeepResult(result=data)
        return {'result': result}
