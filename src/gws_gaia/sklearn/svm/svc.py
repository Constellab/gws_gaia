# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

from gws_core import (BoolParam, InputSpec, OutputSpec, StrParam, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.svm import SVC

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedRegResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# SVCResult
#
# *****************************************************************************


@resource_decorator("SVCResult", hide=True)
class SVCResult(BaseSupervisedRegResult):
    pass

# *****************************************************************************
#
# SVCTrainer
#
# *****************************************************************************


@task_decorator("SVCTrainer", human_name="SVC trainer",
                short_description="Train a C-Support Vector Classifier (SVC) model")
class SVCTrainer(BaseSupervisedTrainer):
    """
    Trainer of a C-Support Vector Classifier (SVC) model. Fit a SVC model according to a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(SVCResult, human_name="result", short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'probability': BoolParam(default_value=False),
        'kernel': StrParam(default_value='rbf')
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return SVC(probability=params["probability"], kernel=params["kernel"])

    @classmethod
    def create_result_class(cls) -> Type[SVCResult]:
        return SVCResult

# *****************************************************************************
#
# SVCPredictor
#
# *****************************************************************************


@task_decorator("SVCPredictor", human_name="SVC predictor",
                short_description="Predict table class labels using a trained C-Support Vector Classifier (SVC) model")
class SVCPredictor(BaseSupervisedPredictor):
    """
    Predictor of a C-Support Vector Classifier (SVC) model. Predict class labels of a table with a trained SVC model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html for more details.
    """
    input_specs = InputSpecs({
        'table': InputSpec(Table, human_name="Table", short_description="The input table"),
        'learned_model': InputSpec(SVCResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
