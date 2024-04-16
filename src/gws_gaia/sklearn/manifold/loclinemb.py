

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.manifold import LocallyLinearEmbedding

from ..base.base_unsup import BaseUnsupervisedResult, BaseUnsupervisedTrainer

# *****************************************************************************
#
# LocallyLinearEmbeddingResult
#
# *****************************************************************************


@resource_decorator("LocallyLinearEmbeddingResult", hide=True)
class LocallyLinearEmbeddingResult(BaseUnsupervisedResult):
    pass

# *****************************************************************************
#
# LocallyLinearEmbeddingTrainer
#
# *****************************************************************************


@task_decorator("LocallyLinearEmbeddingTrainer", human_name="LLE trainer",
                short_description="Trainer of a Locally Linear Embedding (LLE) model")
class LocallyLinearEmbeddingTrainer(BaseUnsupervisedTrainer):
    """
    Trainer of a locally linear embedding model. Compute the embedding vectors for a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(LocallyLinearEmbeddingResult,
                                         human_name="result", short_description="The output result")})
    config_specs = {
        'nb_components': IntParam(default_value=2, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return LocallyLinearEmbedding(n_components=params["nb_components"])

    @classmethod
    def create_result_class(cls) -> Type[LocallyLinearEmbeddingResult]:
        return LocallyLinearEmbeddingResult
