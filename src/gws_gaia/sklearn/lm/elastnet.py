# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import Any, Type

import numpy as np
from gws_core import (FloatParam, FloatRField, InputSpec, IntParam, OutputSpec,
                      Table, TechnicalInfo, resource_decorator, task_decorator)
from pandas import DataFrame
from sklearn.linear_model import ElasticNet

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# ElasticNetResult
#
# *****************************************************************************


@resource_decorator("ElasticNetResult", hide=True)
class ElasticNetResult(BaseSupervisedResult):
    """ ElasticNetResult """

    PREDICTION_TABLE_NAME = "Prediction table"
    _r2: int = FloatRField()

    def __init__(self, training_set=None, training_design=None, result=None):
        super().__init__(training_set=training_set, training_design=training_design, result=result)
        if training_set is not None:
            self._create_r2()

    def _create_r2(self) -> float:
        if not self._r2:
            eln = self.get_result()
            training_set = self.get_training_set()
            training_design = self.get_training_design()
            x_true, y_true = TrainingDesignHelper.create_training_matrices(training_set, training_design)
            self._r2 = eln.score(
                X=x_true,
                y=y_true
            )
        technical_info = TechnicalInfo(key='R2', value=self._r2)
        self.add_technical_info(technical_info)

# *****************************************************************************
#
# ElasticNetTrainer
#
# *****************************************************************************


@task_decorator("ElasticNetTrainer", human_name="ElasticNet trainer",
                short_description="Train an ElasticNet model")
class ElasticNetTrainer(BaseSupervisedTrainer):
    """
    Trainer of an elastic net model. Fit model with coordinate descent.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table")}
    output_specs = {'result': OutputSpec(ElasticNetResult, human_name="result", short_description="The output result")}
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'alpha': FloatParam(default_value=1, min_value=0),
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return ElasticNet(alpha=params["alpha"])

    @classmethod
    def create_result_class(cls) -> Type[BaseSupervisedResult]:
        return ElasticNetResult

# *****************************************************************************
#
# ElasticNetPredictor
#
# *****************************************************************************


@task_decorator("ElasticNetPredictor", human_name="Elastic-Net predictor",
                short_description="Predict table targets using a trained Elastic-Net model")
class ElasticNetPredictor(BaseSupervisedPredictor):
    """
    Predictor of a trained elastic net model. Predict from a table using the trained model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for more details
    """
    input_specs = {'table': InputSpec(Table, human_name="Table", short_description="The input table"),
                   'learned_model': InputSpec(ElasticNetResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Table, human_name="result", short_description="The output result")}
    config_specs = {}
