
# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (Resource, ResourceDecorator)

#==============================================================================
#==============================================================================

@ResourceDecorator("Tuple")
class Tuple(Resource):
    def __init__(self, *args, tup: tuple = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = tup
