# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any

from gws_core import (Resource, ResourceRField, ResourceSet, RField,
                      resource_decorator)

# ==============================================================================
# ==============================================================================


@resource_decorator("BaseResource", hide=True)
class BaseResource(Resource):
    _result: Any = RField(default_value=None)
    _training_set: Resource = ResourceRField()

    def __init__(self, training_set=None, result=None):
        super().__init__()
        if training_set is not None:
            self._training_set = training_set
        if result is not None:
            self._result = result

    def get_training_set(self):
        return self._training_set

    def get_result(self):
        return self._result


@resource_decorator("BaseResourceSet", hide=True)
class BaseResourceSet(ResourceSet):
    _result: Any = RField(default_value=None)
    _training_set: Resource = ResourceRField()

    def __init__(self, training_set=None, result=None):
        super().__init__()
        if training_set is not None:
            self._training_set = training_set
        if result is not None:
            self._result = result

    def get_training_set(self):
        return self._training_set

    def get_result(self):
        return self._result
