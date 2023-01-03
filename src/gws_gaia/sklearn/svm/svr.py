# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (ConfigParams, FloatParam, InputSpec, IntParam,
                      OutputSpec, Resource, StrParam, Table, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from numpy import ravel
from pandas import DataFrame
from sklearn.svm import SVR

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# SVRResult
#
# *****************************************************************************


@resource_decorator("SVRResult", hide=True)
class SVRResult(BaseSupervisedResult):
    pass

# *****************************************************************************
#
# SVRTrainer
#
# *****************************************************************************


@task_decorator("SVRTrainer", human_name="SVC trainer",
                short_description="Train a C-Support Vector Regression (SVC) model")
class SVRTrainer(BaseSupervisedTrainer):
    """
    Trainer of a Epsilon-Support Vector Regression (SVR) model. Fit a SVR model according to a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(SVRResult, human_name="result", short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'kernel': StrParam(default_value='rbf')
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return SVR(kernel=params["kernel"])

    @classmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        return SVRResult

# *****************************************************************************
#
# SVRPredictor
#
# *****************************************************************************


@task_decorator("SVRPredictor", human_name="SVR predictor",
                short_description="Predict table targets using a trained Epsilon-Support Vector Regression (SVR) model")
class SVRPredictor(BaseSupervisedPredictor):
    """
    Predictor of a Epsilon-Support Vector Regression (SVR) model. Predict target values of a table with a trained SVR model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html for more details.
    """
    input_specs = {
        'table': InputSpec(Table, human_name="Table", short_description="The input table"),
        'learned_model': InputSpec(SVRResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
