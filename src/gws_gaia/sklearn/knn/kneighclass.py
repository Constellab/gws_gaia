

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.neighbors import KNeighborsClassifier

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedClassResult,
                             BaseSupervisedPredictor, BaseSupervisedTrainer)

# *****************************************************************************
#
# KNNClassifierResult
#
# *****************************************************************************


@resource_decorator("KNNClassifierResult", hide=True)
class KNNClassifierResult(BaseSupervisedClassResult):
    pass

# *****************************************************************************
#
# KNNClassifierTrainer
#
# *****************************************************************************


@task_decorator("KNNClassifierTrainer", human_name="KNN classifier trainer",
                short_description="Train a k-nearest neighbors classifier model")
class KNNClassifierTrainer(BaseSupervisedTrainer):
    """
    Trainer of a k-nearest neighbors classifier. Fit the k-nearest neighbors classifier from the training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(KNNClassifierResult, human_name="result",
                                         short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'nb_neighbors': IntParam(default_value=5, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return KNeighborsClassifier(n_neighbors=params["nb_neighbors"])

    @classmethod
    def create_result_class(cls) -> Type[KNNClassifierResult]:
        return KNNClassifierResult

# *****************************************************************************
#
# KNNClassifierPredictor
#
# *****************************************************************************


@task_decorator("KNNClassifierPredictor", human_name="KNN classifier predictor",
                short_description="Predict table labels using a trained k-nearest neighbors classifier model")
class KNNClassifierPredictor(BaseSupervisedPredictor):
    """
    Predictor of a K-nearest neighbors classifier. Predict the class labels for a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        KNNClassifierResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
