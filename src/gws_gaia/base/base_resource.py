# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any
from gws_core import (Resource, resource_decorator)


#==============================================================================
#==============================================================================

@resource_decorator("BaseResource", hide=True)
class BaseResource(Resource):

    @classmethod
    def from_result(cls, result: Any) -> 'BaseResource':
        resource = cls()
        resource.binary_store['result'] = result
        return resource