# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import Tuple
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("LogisticRegressionResult", hide=True)
class LogisticRegressionResult(BaseResource):
    pass
#==============================================================================
#==============================================================================

@task_decorator("LogisticRegressionTrainer")
class LogisticRegressionTrainer(Task):
    """
    Trainer of a logistic regression classifier. Fit a logistic regression model according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : LogisticRegressionResult}
    config_specs = {
        'inv_reg_strength': FloatParam(default_value=1, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        logreg = LogisticRegression(C=params["inv_reg_strength"])
        logreg.fit(dataset.features.values, ravel(dataset.targets.values))
        result = LogisticRegressionResult.from_result(logreg=logreg)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("LogisticRegressionTester")
class LogisticRegressionTester(Task):
    """
    Tester of a trained logistic regression classifier. Return the mean accuracy on a given dataset for a trained logistic regression classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LogisticRegressionResult}
    output_specs = {'result' : Tuple}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        logreg = learned_model.binary_store['result']
        y = logreg.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = Tuple(tup = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("LogisticRegressionPredictor")
class LogisticRegressionPredictor(Task):
    """
    Predictor of a logistic regression classifier. Predict class labels for samples in a dataset with a trained logistic regression classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LogisticRegressionResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        logreg = learned_model.binary_store['result']
        y = logreg.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}