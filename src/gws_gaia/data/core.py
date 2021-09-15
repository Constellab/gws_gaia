
# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.models import save_model, load_model

from gws_core import (Resource, resource_decorator)

#==============================================================================
#==============================================================================

@resource_decorator("Tuple")
class Tuple(Resource):
    def __init__(self, *args, tup: tuple = None, **kwargs):
        super().__init__(*args, **kwargs)
        #self._data = tup
        if tup is not None:
            if isinstance(tup, KerasModel):
                path = os.path.join(self.binary_store.full_file_dir, "keras_model")
                save_model(tup, path)
                self.binary_store['keras_model_path'] = path
            else:
                self.binary_store["tuple"] = tup

    @property
    def _data(self):
        if 'keras_model_path' in self.binary_store:
            path = self.binary_store['keras_model_path']
            return load_model(path)
        else:
            return self.binary_store["tuple"]