# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator)
from sklearn.neighbors import KNeighborsRegressor

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedRegResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# KNNRegressorResult
#
# *****************************************************************************


@resource_decorator("KNNRegressorResult", hide=True)
class KNNRegressorResult(BaseSupervisedRegResult):
    pass

# *****************************************************************************
#
# KNNRegressorTrainer
#
# *****************************************************************************


@task_decorator("KNNRegressorTrainer", human_name="KNN regressor trainer",
                short_description="Train a k-nearest neighbors regression model")
class KNNRegressorTrainer(BaseSupervisedTrainer):
    """
    Trainer for a k-nearest neighbors regressor. Fit a k-nearest neighbors regressor from a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(KNNRegressorResult, human_name="result",
                                         short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'nb_neighbors': IntParam(default_value=5, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return KNeighborsRegressor(n_neighbors=params["nb_neighbors"])

    @classmethod
    def create_result_class(cls) -> Type[KNNRegressorResult]:
        return KNNRegressorResult

# *****************************************************************************
#
# KNNRegressorPredictor
#
# *****************************************************************************


@task_decorator("KNNRegressorPredictor", human_name="KNN regressor predictor",
                short_description="Predict table targets using a trained k-nearest neighbors regression model")
class KNNRegressorPredictor(BaseSupervisedPredictor):
    """
    Predictor for a k-nearest neighbors regressor. Predict the regression target for a table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        KNNRegressorResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
