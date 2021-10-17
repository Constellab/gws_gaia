# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from sklearn.manifold import LocallyLinearEmbedding

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("LocallyLinearEmbeddingResult", hide=True)
class LocallyLinearEmbeddingResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("LocallyLinearEmbeddingTrainer")
class LocallyLinearEmbeddingTrainer(Task):
    """
    Trainer of a locally linear embedding model. Compute the embedding vectors for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : LocallyLinearEmbeddingResult}
    config_specs = {
        'nb_components': IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        lle = LocallyLinearEmbedding(n_components=params["nb_components"])
        lle.fit(dataset.get_features().values)
        result = LocallyLinearEmbeddingResult(result = lle)
        return {'result': result}