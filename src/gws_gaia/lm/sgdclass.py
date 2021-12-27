# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import SGDClassifier

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from gws_core import Dataset
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# SGDClassifierResult
#
# *****************************************************************************

@resource_decorator("SGDClassifierResult")
class SGDClassifierResult(BaseResource):
    pass

# *****************************************************************************
#
# SGDClassifierTrainer
#
# *****************************************************************************

@task_decorator("SGDClassifierTrainer")
class SGDClassifierTrainer(Task):
    """
    Trainer of a linear classifier with stochastic gradient descent (SGD). Fit a SGD linear classifier with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : SGDClassifierResult}
    config_specs = {
        'loss':StrParam(default_value='hinge'),
        'alpha': FloatParam(default_value=0.0001, min_value=0),
        'max_iter':IntParam(default_value=1000, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        sgdc = SGDClassifier(max_iter=params["max_iter"],alpha=params["alpha"],loss=params["loss"])
        sgdc.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = SGDClassifierResult(result = sgdc)
        return {'result': result}

# *****************************************************************************
#
# SGDClassifierPredictor
#
# *****************************************************************************

@task_decorator("SGDClassifierPredictor")
class SGDClassifierPredictor(Task):
    """
    Predictor of a linear classifier with stochastic gradient descent (SGD). Predict class labels of a dataset with a trained SGD linear classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html for more details.
    """    
    input_specs = {'dataset' : Dataset, 'learned_model': SGDClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        sgdc = learned_model.result
        y = sgdc.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y, 
            row_names=dataset.row_names, 
            column_names=dataset.target_names, 
            target_names=dataset.target_names
        )
        return {'result': result_dataset}