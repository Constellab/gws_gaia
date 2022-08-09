# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)
from sklearn.manifold import LocallyLinearEmbedding

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# LocallyLinearEmbeddingResult
#
# *****************************************************************************


@resource_decorator("LocallyLinearEmbeddingResult", hide=True)
class LocallyLinearEmbeddingResult(BaseResourceSet):
    pass

# *****************************************************************************
#
# LocallyLinearEmbeddingTrainer
#
# *****************************************************************************


@task_decorator("LocallyLinearEmbeddingTrainer", human_name="LLE trainer",
                short_description="Trainer of a Locally Linear Embedding (LLE) model")
class LocallyLinearEmbeddingTrainer(Task):
    """
    Trainer of a locally linear embedding model. Compute the embedding vectors for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(LocallyLinearEmbeddingResult, human_name="result", short_description="The output result")}
    config_specs = {
        'nb_components': IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        lle = LocallyLinearEmbedding(n_components=params["nb_components"])
        lle.fit(dataset.get_features().values)
        result = LocallyLinearEmbeddingResult(training_set=dataset, result=lle)
        return {'result': result}
