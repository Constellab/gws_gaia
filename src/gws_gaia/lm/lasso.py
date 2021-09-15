# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import Lasso

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import Tuple
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("LassoResult", hide=True)
class LassoResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("LassoTrainer")
class LassoTrainer(Task):
    """
    Trainer of a lasso model. Fit a lasso model with a training dataset.

    See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.Lasso.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : LassoResult}
    config_specs = {
        'alpha': FloatParam(default_value=1, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        las = Lasso(alpha=params["alpha"])
        las.fit(dataset.features.values, ravel(dataset.targets.values))
        result = LassoResult.from_result(result=las)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("LassoTester")
class LassoTester(Task):
    """
    Tester of a trained lasso model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained lasso model.
    
    See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.Lasso.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LassoResult}
    output_specs = {'result' : Tuple}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        las = learned_model.binary_store['result']
        y = las.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = Tuple(tup = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("LassoPredictor")
class LassoPredictor(Task):
    """
    Predictor of a lasso model. Predict target values from a dataset with a trained lasso model.

    See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.Lasso.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LassoResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        las = learned_model.binary_store['result']
        y = las.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}