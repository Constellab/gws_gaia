# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# LogisticRegressionResult
#
# *****************************************************************************


@resource_decorator("LogisticRegressionResult", hide=True)
class LogisticRegressionResult(BaseResourceSet):
    pass

# *****************************************************************************
#
# LogisticRegressionTrainer
#
# *****************************************************************************


@task_decorator("LogisticRegressionTrainer", human_name="Logistic regression trainer",
                short_description="Train a logistic regression classifier")
class LogisticRegressionTrainer(Task):
    """
    Trainer of a logistic regression classifier. Fit a logistic regression model according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(LogisticRegressionResult, human_name="result", short_description="The output result")}
    config_specs = {
        'inv_reg_strength': FloatParam(default_value=1, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        logreg = LogisticRegression(C=params["inv_reg_strength"])
        logreg.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = LogisticRegressionResult(training_set=dataset, result=logreg)
        return {'result': result}

# *****************************************************************************
#
# LogisticRegressionPredictor
#
# *****************************************************************************


@task_decorator("LogisticRegressionPredictor", human_name="Logistic regression predictor",
                short_description="Predict dataset labels using a trained logistic regression classifier")
class LogisticRegressionPredictor(Task):
    """
    Predictor of a logistic regression classifier. Predict class labels for samples in a dataset with a trained logistic regression classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(LogisticRegressionResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        logreg = learned_model.get_result()
        y = logreg.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
