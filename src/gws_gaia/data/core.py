
# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
from typing import Any
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.models import save_model, load_model
from dill import load, dump

from gws_core import Serializer, resource_decorator
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("GenericResult")
class GenericResult(BaseResource):
    def __init__(self, *args, tup: tuple = None, **kwargs):
        super().__init__(*args, **kwargs)

    def get_result(self) -> Any:
        return self.binary_store.load('result', Serializer.load)

    @classmethod
    def from_result(cls, result: Any) -> 'BaseResource':
        resource = cls()
        resource.binary_store.dump('result', result, Serializer.dump)
        return resource