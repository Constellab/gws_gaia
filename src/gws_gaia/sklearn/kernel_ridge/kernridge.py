# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (FloatParam, InputSpec, OutputSpec, StrParam, Table,
                      resource_decorator, task_decorator)
from sklearn.kernel_ridge import KernelRidge

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedRegResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# KernelRidgeResult
#
# *****************************************************************************


@resource_decorator("KernelRidgeResult", hide=True)
class KernelRidgeResult(BaseSupervisedRegResult):
    pass

# *****************************************************************************
#
# KernelRidgeTrainer
#
# *****************************************************************************


@task_decorator("KernelRidgeTrainer", human_name="Kernel-Ridge trainer",
                short_description="Train a kernel ridge regression model")
class KernelRidgeTrainer(BaseSupervisedTrainer):
    """
    Trainer of a kernel ridge regression model. Fit a kernel ridge regression model with a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(KernelRidgeResult, human_name="result", short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'gamma': FloatParam(default_value=None),
        'kernel': StrParam(default_value='linear')
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return KernelRidge(gamma=params["gamma"], kernel=params["kernel"])

    @classmethod
    def create_result_class(cls) -> Type[KernelRidgeResult]:
        return KernelRidgeResult

# *****************************************************************************
#
# KernelRidgePredictor
#
# *****************************************************************************


@task_decorator("KernelRidgePredictor", human_name="Kernel-Ridge predictor",
                short_description="Predict table targets using a trained kernel ridge regression model")
class KernelRidgePredictor(BaseSupervisedPredictor):
    """
    Predictor of a kernel ridge regression model. Predict a regression target from a table with a trained kernel ridge regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html for more details.
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        KernelRidgeResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
