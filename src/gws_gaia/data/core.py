
# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
from typing import Any
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.models import save_model, load_model
from dill import load, dump

from gws_core import Serializer, resource_decorator, RField
from ..base.base_resource import BaseResource

@resource_decorator("GenericResult", hide=True)
class GenericResult(BaseResource):
    result: Any = RField(default_value=None)

    def __init__(self, result=None):
        super().__init__()
        if result is not None:
            self.result = result