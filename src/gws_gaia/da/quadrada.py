# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from pandas import DataFrame
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# QDAResult
#
# *****************************************************************************


@resource_decorator("QDAResult", human_name="QDA result", hide=True)
class QDAResult(BaseResourceSet):
    pass

# *****************************************************************************
#
# QDATrainer
#
# *****************************************************************************


@task_decorator("QDATrainer", human_name="QDA trainer",
                short_description="Train a Quadratic Discriminant Analysis (QDA) model")
class QDATrainer(Task):
    """
    Trainer of quadratic discriminant analysis model. Fit a quadratic discriminant analysis model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(QDAResult, human_name="result", short_description="The output result")}
    config_specs = {
        'reg_param': FloatParam(default_value=0),
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        qda = QuadraticDiscriminantAnalysis(reg_param=params["reg_param"])
        qda.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = QDAResult(training_set=dataset, result=qda)
        return {'result': result}

# *****************************************************************************
#
# QDAPredictor
#
# *****************************************************************************


@task_decorator("QDAPredictor", human_name="QDA predictor",
                short_description="Predict class labels using a Quadratic Discriminant Analysis (QDA) model")
class QDAPredictor(Task):
    """
    Predictor of quadratic discriminant analysis model. Predic class labels of a dataset with a trained quadratic discriminant analysis model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(QDAResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        qda = learned_model.get_result()
        y = qda.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=DataFrame(y),
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names,
        )
        return {'result': result_dataset}
