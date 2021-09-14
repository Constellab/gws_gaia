# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.core import Tuple
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("QDAResult")
class QDAResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("QDAResult")
class QDATrainer(Task):
    """
    Trainer of quadratic discriminant analysis model. Fit a quadratic discriminant analysis model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : QDAResult}
    config_specs = {
        'reg_param': FloatParam(default_value=0),
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        qda = QuadraticDiscriminantAnalysis(reg_param=params["reg_param"])
        qda.fit(dataset.features.values, ravel(dataset.targets.values))
        result = QDAResult.from_result(result=qda)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("QDATester")
class QDATester(Task):
    """
    Tester of a trained quadratic discriminant analysis model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained quadratic discriminant analysis model.
    
    See ttps://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': QDAResult}
    output_specs = {'result' : Tuple}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        qda = learned_model.binary_store['result']
        y = qda.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = Tuple(tup = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("QDAPredictor")
class QDAPredictor(Task):
    """
    Predictor of quadratic discriminant analysis model. Predic class labels of a dataset with a trained quadratic discriminant analysis model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': QDAResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        qda = learned_model.binary_store['result']
        y = qda.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}