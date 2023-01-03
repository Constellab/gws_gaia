# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, StrParam, Table,
                      resource_decorator, task_decorator)
from sklearn.cluster import AgglomerativeClustering

from ..base.base_unsup import BaseUnsupervisedTrainer
from .base_clust_result import BaseClusteringResult

# *****************************************************************************
#
# AgglomerativeClusteringResult
#
# *****************************************************************************


@resource_decorator("AgglomerativeClusteringResult", hide=True)
class AgglomerativeClusteringResult(BaseClusteringResult):
    """ AgglomerativeClusteringResult """

# *****************************************************************************
#
# AgglomerativeClusteringTrainer
#
# *****************************************************************************


@task_decorator("AgglomerativeClusteringTrainer", human_name="Agglomerative clustering trainer",
                short_description="Train a the hierarchical clustering model")
class AgglomerativeClusteringTrainer(BaseUnsupervisedTrainer):
    """ Trainer of the hierarchical clustering. Fits the hierarchical clustering from features, or distance matrix.
@
    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(AgglomerativeClusteringResult,
                                         human_name="result", short_description="The output result")}
    config_specs = {
        "nb_clusters": IntParam(default_value=2, min_value=0),
        "linkage": StrParam(default_value="ward", allowed_values=["ward", "complete", "average", "single"]),
        "affinity":
        StrParam(
            default_value="euclidean", allowed_values=["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
            short_description="Metric used to compute the linkage."), }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Type[Any]:
        return AgglomerativeClustering(n_clusters=params["nb_clusters"], linkage=params["linkage"])

    @classmethod
    def create_result_class(cls) -> Type[AgglomerativeClusteringResult]:
        return AgglomerativeClusteringResult
