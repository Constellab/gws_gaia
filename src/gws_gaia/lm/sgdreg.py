# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import SGDRegressor

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from gws_core import Dataset
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# SGDRegressorResult
#
# *****************************************************************************

@resource_decorator("SGDRegressorResult", hide=True)
class SGDRegressorResult(BaseResource):
    pass

# *****************************************************************************
#
# SGDRegressorTrainer
#
# *****************************************************************************

@task_decorator("SGDRegressorTrainer")
class SGDRegressorTrainer(Task):
    """
    Trainer of a linear regressor with stochastic gradient descent (SGD). Fit a SGD linear regressor with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for more details.
    """    
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : SGDRegressorResult}
    config_specs = {
        'loss':StrParam(default_value='squared_loss'),
        'alpha': FloatParam(default_value=0.0001, min_value=0),
        'max_iter':IntParam(default_value=1000, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        sgdr = SGDRegressor(max_iter=params["max_iter"],alpha=params["alpha"],loss=params["loss"])
        sgdr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = SGDRegressorResult(result = sgdr)
        return {'result': result}

# *****************************************************************************
#
# SGDRegressorPredictor
#
# *****************************************************************************

@task_decorator("SGDRegressorPredictor")
class SGDRegressorPredictor(Task):
    """
    Predictor of a linear regressor with stochastic gradient descent (SGD). Predict target values of a dataset with a trained SGD linear regressor.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for more details.
    """        
    input_specs = {'dataset' : Dataset, 'learned_model': SGDRegressorResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        sgdr = learned_model.result
        y = sgdr.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y, 
            row_names=dataset.row_names, 
            column_names=dataset.target_names, 
            target_names=dataset.target_names
        )
        return {'result': result_dataset}