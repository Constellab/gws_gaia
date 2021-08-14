# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import numpy as np
from numpy import ravel
import tensorflow as tf
from tensorflow.keras import Model as KerasModel
from pandas import DataFrame

from gws_core import (Process, Resource, ProcessDecorator, ResourceDecorator)
from ..data.core import Tuple
from ..data.dataset import Dataset
from .data import Tensor, DeepModel


#==================================================================================
#==================================================================================

@ProcessDecorator("DeepModelerBuilder")
class DeepModelerBuilder(Process):
    """
    Build the model from layers specifications
    """
    input_specs = {'inputs' : Tensor, 'outputs': Tensor}
    output_specs = {'result' : DeepModel}
    config_specs = {
    }

    async def task(self):
        x = self.input['inputs']
        y = self.input['outputs']
        x1 = x._data
        y1 = y._data
        z = KerasModel(inputs=x1, outputs=y1)
        
        t = self.output_specs["result"]
        result = t(model=z)
        self.output['result'] = result

#==================================================================================
#==================================================================================

@ProcessDecorator("DeepModelerCompiler")
class DeepModelerCompiler(Process):
    """
    Configures the model for training.

    See https://keras.io/api/models/model_training_apis/ for more details
    """
    input_specs = {'builded_model' : DeepModel}
    output_specs = {'result' : DeepModel}
    config_specs = {
        'optimizer': {"type": 'str', "default": 'rmsprop'},
        'loss': {"type": 'str', "default": ''},
        'metrics': {"type": 'str', "default": ''}
    }

    async def task(self):
        x = self.input['builded_model']
        model = x._data
        model.compile(optimizer=self.get_param("optimizer"), loss=self.get_param("loss"), metrics=self.get_param("metrics"))
        t = self.output_specs["result"]
        result = t(model=model)
        self.output['result'] = result

#==================================================================================
#==================================================================================

@ProcessDecorator("DeepModelerTrainer")
class DeepModelerTrainer(Process):
    """
    Trainer of a model on a dataset

    See https://keras.io/api/models/model_training_apis/ for more details
    """
    input_specs = {'dataset' : Tuple, 'compiled_model': DeepModel}
    output_specs = {'result' : DeepModel}
    config_specs = {
        'batch_size': {"type": 'int', "default": 32, "min": 0},
        'epochs': {"type": 'int', "default": 1, "min": 0},
        'validation_split': {"type": 'float', "default": 0.1, "min": 0},
    }

    async def task(self):
        x = self.input['compiled_model']
        model = x._data
        y = self.input['dataset']
        data = y._data

        (x_train, y_train), (_, _) = data
        model.fit(x_train, y_train, batch_size=self.get_param("batch_size"), epochs=self.get_param("epochs"), validation_split=self.get_param("validation_split"))
        
        t = self.output_specs["result"]
        result = t(model=model)
        self.output['result'] = result

#==================================================================================
#==================================================================================

@ProcessDecorator("DeepModelerTester")
class DeepModelerTester(Process):
    """
    Tester of a model on a test dataset.

    See https://keras.io/api/models/model_training_apis/ for more details
    """
    input_specs = {'dataset' : Tuple, 'trained_model': DeepModel}
    output_specs = {'result' : Tuple}
    config_specs = {
        'verbosity_mode': {"type": 'int', "default": 0},
    }

    async def task(self):
        x = self.input['trained_model']
        model = x._data
        y = self.input['dataset']
        data = y._data

        (_, _), (x_test, y_test) = data
        score = model.evaluate(x_test, y_test, verbose=self.get_param('verbosity_mode'))
        print(score)
        t = self.output_specs["result"]
        result = t(tup=score)
        self.output['result'] = result

#==================================================================================
#==================================================================================

@ProcessDecorator("DeepModelerPredictor")
class DeepModelerPredictor(Process):
    """
    Predictor of a trained model from a dataset. Generates output predictions for the input samples.

    See https://keras.io/api/models/model_training_apis/ for more details
    """
    input_specs = {'dataset' : Tuple, 'trained_model': DeepModel}
    output_specs = {'result' : Tuple}
    config_specs = {
        'verbosity_mode': {"type": 'int', "default": 0},
    }

    async def task(self):
        x = self.input['trained_model']
        model = x._data
        y = self.input['dataset']
        data = y._data

        result = model.predict(data, verbose=self.get_param('verbosity_mode'))

        t = self.output_specs["result"]
        result = t(tup=result)
        self.output['result'] = result