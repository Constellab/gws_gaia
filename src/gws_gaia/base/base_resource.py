

from typing import Any, Dict

from gws_core import (Resource, ResourceRField, ResourceSet, RField,
                      resource_decorator)

@resource_decorator("BaseResourceSet", hide=True)
class BaseResourceSet(ResourceSet):
    """ The BaseResourceSet """

    _result: Any = RField(default_value=None)
    _training_set: Resource = ResourceRField()
    _training_design: Dict = RField()

    def __init__(self, training_set=None, training_design=None, result=None):
        super().__init__()
        if training_set is not None:
            self._training_set = training_set
        if training_design is not None:
            self._training_design = training_design
        if result is not None:
            self._result = result

    def get_training_set(self):
        """ Get the training set """
        return self._training_set

    def get_training_design(self):
        return self._training_design

    def get_result(self):
        """ Get the result """
        return self._result
