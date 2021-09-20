# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from numpy import ravel
from pandas import DataFrame
from sklearn.tree import DecisionTreeRegressor

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("DecisionTreeRegressorResult")
class DecisionTreeRegressorResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("DecisionTreeRegressorTrainer")
class DecisionTreeRegressorTrainer(Task):
    """ Trainer of a decision tree regressor. Build a decision tree regressor from a training set   

    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : DecisionTreeRegressorResult}
    config_specs = {
        'max_depth':IntParam(default_value=None, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        dtr = DecisionTreeRegressor(max_depth=params["max_depth"])
        dtr.fit(dataset.features.values, ravel(dataset.targets.values))
        result = DecisionTreeRegressorResult(result = dtr)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("DecisionTreeRegressorTester")
class DecisionTreeRegressorTester(Task):
    """
    Tester of a trained decision tree regressor. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained decision tree regressor.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': DecisionTreeRegressorResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        dtr = learned_model.result
        y = dtr.score(dataset.features.values,ravel(dataset.targets.values))
        z = tuple([y])
        result_dataset = GenericResult(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("DecisionTreeRegressorPredictor")
class DecisionTreeRegressorPredictor(Task):
    """ Predictor of a trained decision tree regressor. Predict class or regression value for the dataset. For a classification model, the predicted class for each sample in the dataset is returned. For a regression model, the predicted value based on the dataset is returned.

    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': DecisionTreeRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        dtr = learned_model.result
        y = dtr.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}