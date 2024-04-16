

from typing import Any, Type

from gws_core import (InputSpec, IntParam, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.ensemble import RandomForestRegressor

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedRegResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# RandomForestRegressorResult
#
# *****************************************************************************


@resource_decorator("RandomForestRegressorResult", hide=True)
class RandomForestRegressorResult(BaseSupervisedRegResult):
    """ RandomForestRegressorResult """

# *****************************************************************************
#
# RandomForestRegressorTrainer
#
# *****************************************************************************


@task_decorator("RandomForestRegressorTrainer", human_name="Random-Forest regressor trainer",
                short_description="Train a random forest regression model")
class RandomForestRegressorTrainer(BaseSupervisedTrainer):
    """
    Trainer of a random forest regressor. Build a forest of trees from a training table.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(RandomForestRegressorResult,
                                         human_name="result", short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'nb_estimators': IntParam(default_value=100, min_value=0)
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return RandomForestRegressor(n_estimators=params["nb_estimators"])

    @classmethod
    def create_result_class(cls) -> Type[RandomForestRegressorResult]:
        return RandomForestRegressorResult

# *****************************************************************************
#
# RandomForestRegressorPredictor
#
# *****************************************************************************


@task_decorator("RandomForestRegressorPredictor", human_name="Random-Forest regression predictor",
                short_description="Predict table targets using a trained Random forest regression model")
class RandomForestRegressorPredictor(BaseSupervisedPredictor):
    """
    Predictor of a random forest regressor. Predict regression target of a table with a trained random forest regressor.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html for more details.
    """
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"), 'learned_model': InputSpec(
        RandomForestRegressorResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
