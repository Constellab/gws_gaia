

from typing import Any, Type

from gws_core import (FloatParam, InputSpec, OutputSpec, Table,
                      resource_decorator, task_decorator, InputSpecs, OutputSpecs)
from sklearn.linear_model import ElasticNet

from ...base.helper.training_design_helper import TrainingDesignHelper
from ..base.base_sup import (BaseSupervisedPredictor, BaseSupervisedRegResult,
                             BaseSupervisedTrainer)

# *****************************************************************************
#
# ElasticNetResult
#
# *****************************************************************************


@resource_decorator("ElasticNetResult", hide=True)
class ElasticNetResult(BaseSupervisedRegResult):
    """ ElasticNetResult """

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
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table")})
    output_specs = OutputSpecs({'result': OutputSpec(ElasticNetResult, human_name="result", short_description="The output result")})
    config_specs = {
        'training_design': TrainingDesignHelper.create_training_design_param_set(),
        'alpha': FloatParam(default_value=1, min_value=0),
    }

    @classmethod
    def create_sklearn_trainer_class(cls, params) -> Any:
        return ElasticNet(alpha=params["alpha"])

    @classmethod
    def create_result_class(cls) -> Type[ElasticNetResult]:
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
    input_specs = InputSpecs({'table': InputSpec(Table, human_name="Table", short_description="The input table"),
                   'learned_model': InputSpec(ElasticNetResult, human_name="Learned model", short_description="The input model")})
    output_specs = OutputSpecs({'result': OutputSpec(Table, human_name="result", short_description="The output result")})
    config_specs = {}
