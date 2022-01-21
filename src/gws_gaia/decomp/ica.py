# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, IntParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from sklearn.decomposition import FastICA

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# ICAResult
#
# *****************************************************************************


@resource_decorator("ICAResult", hide=True)
class ICAResult(BaseResource):
    pass

# *****************************************************************************
#
# ICATrainer
#
# *****************************************************************************


@task_decorator("ICATrainer", human_name="ICA trainer",
                short_description="Train an Independant Component Analysis (ICA) model")
class ICATrainer(Task):
    """
    Trainer of an Independant Component Analysis (ICA) model. Fit a model of ICA to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.ICA.html#sklearn.decomposition.ICA.fit for more details.
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': ICAResult}
    config_specs = {
        'nb_components': IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        ica = FastICA(n_components=params["nb_components"])
        ica.fit(dataset.get_features().values)
        result = ICAResult(result=ica)
        return {'result': result}
