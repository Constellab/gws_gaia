# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import LinearRegression

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import Tuple
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("LinearRegressionResult", hide=True)
class LinearRegressionResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("LinearRegressionTrainer")
class LinearRegressionTrainer(Task):
    """
    Trainer fo a linear regression model. Fit a linear regression model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : LinearRegressionResult}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        lir = LinearRegression()
        lir.fit(dataset.features.values, ravel(dataset.targets.values))
        result = LinearRegressionResult.from_result(result=lir)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("LinearRegressionTester")
class LinearRegressionTester(Task):
    """
    Tester of a trained linear regression model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained linear regression model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LinearRegressionResult}
    output_specs = {'result' : Tuple}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lir = learned_model.binary_store['result']
        y = lir.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = Tuple(tup = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("LinearRegressionPredictor")
class LinearRegressionPredictor(Task):
    """
    Predictor of a linear regression model. Predict target values of a dataset with a trained linear regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LinearRegressionResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lir = learned_model.binary_store['result']
        y = lir.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}