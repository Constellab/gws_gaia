# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator)
from sklearn.ensemble import AdaBoostRegressor

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedRegResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# AdaBoostRegressorResult
#
# *****************************************************************************


@resource_decorator("AdaBoostRegressorResult", hide=True)
class AdaBoostRegressorResult(BaseSupervisedRegResult):
    pass

# *****************************************************************************
#
# AdaBoostRegressorTrainer
#
# *****************************************************************************


@task_decorator("AdaBoostRegressorTrainer", human_name="AdaBoost regression trainer",
                short_description="Train an AdaBoost regression model")
class AdaBoostRegressorTrainer(BaseSupervisedTrainer):
    """
    Trainer of an Adaboost regressor. This process build a boosted regressor from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(AdaBoostRegressorResult, human_name="result",
                                         short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'nb_estimators': IntParam(default_value=50, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Type[Any]:
        return AdaBoostRegressor(n_estimators=params["nb_estimators"])

    @classmethod
    def create_result_class(cls) -> Type[AdaBoostRegressorResult]:
        return AdaBoostRegressorResult


# *****************************************************************************
#
# AdaBoostRegressorPredictor
#
# *****************************************************************************


@task_decorator("AdaBoostRegressorPredictor", human_name="AdaBoost regression predictor",
                short_description="Predict table targets using a trained AdaBoost regression model")
class AdaBoostRegressorPredictor(BaseSupervisedPredictor):
    """
    Predictor of a trained Adaboost regressor. The predicted regression value of an input sample is computed as the weighted median
    prediction of the classifiers in the ensemble.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        AdaBoostRegressorResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
