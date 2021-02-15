
# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


# from gws.settings import Settings
# from gws.view import HTMLViewTemplate, JSONViewTemplate, PlainTextViewTemplate
# from gws.model import Resource, HTMLViewModel, JSONViewModel
# from gws.controller import Controller
#
# class SimpleData(Resource):
#     pass
#
# class SimpleDataHTMLViewModel(HTMLViewModel):
#     template = HTMLViewTemplate("Value: {{vmodel.model.data}}")

from gws.model import Resource, Process
from tensorflow.python.framework.ops import Tensor as Kerastensor
from tensorflow.keras import Model as KerasModel
from gws.logger import Error
import tensorflow as tf

#==============================================================================
#==============================================================================

class Tuple(Resource):
    def __init__(self, *args, tup: tuple = None, **kwargs):
        super().__init__(*args, **kwargs)
        #self.kv_store['tensor'] = tensor
        self._data = tup

#==============================================================================
#==============================================================================

class Tensor(Resource):
    def __init__(self, *args, tensor: Kerastensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        #self.kv_store['tensor'] = tensor
        self._data = tensor

#==============================================================================
#==============================================================================

class Model(Resource):
    def __init__(self, *args, model: KerasModel = None,  **kwargs):
        super().__init__(*args, **kwargs)
        
        if not isinstance(model, KerasModel):
            raise Error("Model", "__init__", f"The model must an instance of tensorflow.keras.Model. The given model is {model}")

        self._data = model

#==============================================================================
#==============================================================================

class InputConverter(Process):
    input_specs = {}
    output_specs = {'result' : Tensor}
    config_specs = {
        'input_shape': {"type": 'list'}
    }

    async def task(self):
        input_shape = tuple(self.get_param('input_shape'))
        y = tf.keras.Input(shape=input_shape)
        
        t = self.output_specs["result"]
        result = t(tensor=y)
        self.output['result'] = result