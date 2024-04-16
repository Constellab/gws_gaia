

from typing import Any, Type

from gws_core import (ConfigParams, InputSpec, IntParam, OutputSpec, Table,
                      TaskInputs, TaskOutputs, resource_decorator,
                      task_decorator, InputSpecs, OutputSpecs)
from pandas import DataFrame
from sklearn.cluster import KMeans

from ..base.base_unsup import (BaseUnsupervisedPredictor,
                               BaseUnsupervisedTrainer)
from .base_clust_result import BaseClusteringResult

# *****************************************************************************
#
# KMeansResult
#
# *****************************************************************************


@resource_decorator("KMeansResult", hide=True)
class KMeansResult(BaseClusteringResult):
    """ KMeansResult """

# *****************************************************************************
#
# KMeansTrainer
#
# *****************************************************************************


@task_decorator("KMeansTrainer", human_name="KMeans trainer", short_description="Train a K-Means clustering model")
class KMeansTrainer(BaseUnsupervisedTrainer):
    """
    Trainer of a trained k-means clustering model. Compute a k-means clustering from a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(KMeansResult, human_name="result", short_description="The output result")})
    config_specs = {
        'nb_clusters': IntParam(default_value=2, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Type[Any]:
        return KMeans(n_clusters=params["nb_clusters"])

    @classmethod
    def create_result_class(cls) -> Type[KMeansResult]:
        return KMeansResult

# *****************************************************************************
#
# KMeansPredictor
#
# *****************************************************************************


@task_decorator("KMeansPredictor", human_name="KMeans predictor",
                short_description="Predict the closest cluster each sample using a K-Means model")
class KMeansPredictor(BaseUnsupervisedPredictor):
    """
    Predictor of a K-means clustering model. Predict the closest cluster of each sample in a table belongs to.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more details.
    """
    input_specs = InputSpecs({
        'table': InputSpec(Table, human_name="Table", short_description="The input table"),
        'learned_model': InputSpec(KMeansResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}

    def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        table = inputs['table']
        learned_model = inputs['learned_model']
        kmeans = learned_model.get_result()
        y = kmeans.predict(table.get_data().values)
        result_dataset = Table(
            data=DataFrame(y),
            row_names=table.row_names,
            column_names=["label"],
        )
        return {'result': result_dataset}
