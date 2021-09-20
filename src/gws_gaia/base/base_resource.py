# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any
from gws_core import (Resource, resource_decorator, RField)


#==============================================================================
#==============================================================================

@resource_decorator("BaseResource", hide=True)
class BaseResource(Resource):
    result: Any = RField(default_value=None)

    def __init__(self, result=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if result is not None:
            self.result = result
