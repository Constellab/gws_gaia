
# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any

from gws_core import RField, resource_decorator

from ..base.base_resource import BaseResource


@resource_decorator("GenericResult", hide=True, deprecated_since='0.3.1',
                    deprecated_message="Use DeepResult instead")
class GenericResult(BaseResource):
    _result: Any = RField(default_value=None)

    def __init__(self, result=None):
        super().__init__()
        if result is not None:
            self._result = result
