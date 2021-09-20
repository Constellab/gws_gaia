# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from sklearn.decomposition import FastICA

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("ICAResult", hide=True)
class ICAResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("ICATrainer")
class ICATrainer(Task):
    """
    Trainer of an Independant Component Analysis (ICA). Fit a model of ICA to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.ICA.html#sklearn.decomposition.ICA.fit for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : ICAResult}
    config_specs = {
        'nb_components':IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        ica = FastICA(n_components=params["nb_components"])
        ica.fit(dataset.features.values)
        result = ICAResult(result = ica)
        return {'result': result}