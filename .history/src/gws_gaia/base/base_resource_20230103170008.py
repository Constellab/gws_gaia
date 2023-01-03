# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Dict

from gws_core import (Dataset, Resource, ResourceRField, ResourceSet, RField,
                      Table, resource_decorator)


@resource_decorator("BaseResource", hide=True)
class BaseResource(Resource):
    """ The BaseResource """

    _result: Any = RField(default_value=None)
    _training_set: Resource = ResourceRField()
    _trainig_set_interface = None

    def __init__(self, training_set=None, result=None):
        super().__init__()
        if training_set is not None:
            self._training_set = training_set
        if result is not None:
            self._result = result

    def get_training_set(self):
        """ Get the training set """
        if not isinstance(self._training_set, Dataset):
            if self._trainig_set_interface is None:
                self._trainig_set_interface = Dataset(
                    data=self._training_set.get_data(),
                    meta=self._training_set.get_meta()
                )
            return self._trainig_set_interface
        else:
            return self._training_set

    def get_result(self):
        """ Get the result """
        return self._result


@resource_decorator("BaseResourceSet", hide=True)
class BaseResourceSet(ResourceSet):
    """ The BaseResourceSet """

    _result: Any = RField(default_value=None)
    _training_set: Resource = ResourceRField()
    _training_design: Dict = DictRField()
    _trainig_set_interface = None

    def __init__(self, training_set=None, training_design=None, result=None):
        super().__init__()
        if training_set is not None:
            self._training_set = training_set
        if training_design is not None:
            self._training_set = training_design
        if result is not None:
            self._result = result

    def get_training_set(self):
        """ Get the training set """
        if not isinstance(self._training_set, Dataset):
            if self._trainig_set_interface is None:
                self._trainig_set_interface = Dataset(
                    data=self._training_set.get_data(),
                    meta=self._training_set.get_meta()
                )
            return self._trainig_set_interface
        else:
            return self._training_set

    def get_result(self):
        """ Get the result """
        return self._result
