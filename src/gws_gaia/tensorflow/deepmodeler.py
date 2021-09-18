# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from numpy import ravel
import tensorflow as tf
from tensorflow.keras import Model as KerasModel
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.core import GenericResult
from ..data.dataset import Dataset
from .data import Tensor, DeepModel


#==================================================================================
#==================================================================================

@task_decorator("DeepModelerBuilder")
class DeepModelerBuilder(Task):
    """
    Build the model from layers specifications
    """
    input_specs = {'inputs' : Tensor, 'outputs': Tensor}
    output_specs = {'result' : DeepModel}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['inputs']
        y = inputs['outputs']
        x1 = x.get_result()
        y1 = y.get_result()

        print("xxxx")
        print(x1)
        print(y1)

        z = KerasModel(inputs=x1, outputs=y1)
        result = DeepModel(model=z)
        return {'result': result}

#==================================================================================
#==================================================================================

@task_decorator("DeepModelerCompiler")
class DeepModelerCompiler(Task):
    """
    Configures the model for training.

    See https://keras.io/api/models/model_training_apis/ for more details
    """
    input_specs = {'builded_model' : DeepModel}
    output_specs = {'result' : DeepModel}
    config_specs = {
        'optimizer':StrParam(default_value='rmsprop'),
        'loss':StrParam(default_value=''),
        'metrics':StrParam(default_value='')
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['builded_model']
        model = x.get_result()
        model.compile(optimizer=params["optimizer"], loss=params["loss"], metrics=params["metrics"])
        result = DeepModel(model=model)
        return {'result': result}

#==================================================================================
#==================================================================================

@task_decorator("DeepModelerTrainer")
class DeepModelerTrainer(Task):
    """
    Trainer of a model on a dataset

    See https://keras.io/api/models/model_training_apis/ for more details
    """
    input_specs = {'dataset' : GenericResult, 'compiled_model': DeepModel}
    output_specs = {'result' : DeepModel}
    config_specs = {
        'batch_size':IntParam(default_value=32, min_value=0),
        'epochs':IntParam(default_value=1, min_value=0),
        'validation_split': FloatParam(default_value=0.1, min_value=0),
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['compiled_model']
        model = x.get_result()
        y = inputs['dataset']
        data = y.get_result()
        (x_train, y_train), (_, _) = data
        model.fit(x_train, y_train, batch_size=params["batch_size"], epochs=params["epochs"], validation_split=params["validation_split"])
        result = DeepModel(model=model)
        return {'result': result}

#==================================================================================
#==================================================================================

@task_decorator("DeepModelerTester")
class DeepModelerTester(Task):
    """
    Tester of a model on a test dataset.

    See https://keras.io/api/models/model_training_apis/ for more details
    """
    input_specs = {'dataset' : GenericResult, 'trained_model': DeepModel}
    output_specs = {'result' : GenericResult}
    config_specs = {
        'verbosity_mode':IntParam(default_value=0),
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['trained_model']
        model = x.get_result()
        y = inputs['dataset']
        data = y.get_result()
        (_, _), (x_test, y_test) = data
        score = model.evaluate(x_test, y_test, verbose=params['verbosity_mode'])
        result = GenericResult.from_result(result=score)
        return {'result': result}

#==================================================================================
#==================================================================================

@task_decorator("DeepModelerPredictor")
class DeepModelerPredictor(Task):
    """
    Predictor of a trained model from a dataset. Generates output predictions for the input samples.

    See https://keras.io/api/models/model_training_apis/ for more details
    """
    input_specs = {'dataset' : GenericResult, 'trained_model': DeepModel}
    output_specs = {'result' : GenericResult}
    config_specs = {
        'verbosity_mode':IntParam(default_value=0),
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        x = inputs['trained_model']
        model = x.get_result()
        y = inputs['dataset']
        data = y.get_result()
        result = model.predict(data, verbose=params['verbosity_mode'])
        result = GenericResult.from_result(result=result)
        return {'result': result}