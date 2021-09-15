# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from sklearn.ensemble import AdaBoostClassifier
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.core import Tuple
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

@resource_decorator("AdaBoostClassifierResult", hide=True)
class AdaBoostClassifierResult(BaseResource):
    """AdaBoostClassifierResult"""
    pass

#==============================================================================
#==============================================================================

@task_decorator("AdaBoostClassifierTrainer")
class AdaBoostClassifierTrainer(Task):
    """
    Trainer of an Adaboost classifier. This process builds a boosted classifier from a training set.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : AdaBoostClassifierResult}
    config_specs = {
        'nb_estimators': IntParam(default_value=50, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        abc = AdaBoostClassifier(n_estimators=params["nb_estimators"])
        abc.fit(dataset.features.values, ravel(dataset.targets.values))
        result = AdaBoostClassifierResult.from_result(abc)
        return {"result" : result}

#==============================================================================
#==============================================================================

@task_decorator("AdaBoostClassifierTester")
class AdaBoostClassifierTester(Task):
    """
    Tester of a trained Adaboost classifier. Return the mean accuracy on a given test data and labels for a trained Adaboost classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': AdaBoostClassifierResult}
    output_specs = {'result' : Tuple}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']

        print(inputs)
        
        abc = learned_model.get_result()
        y = abc.score(dataset.features.values,ravel(dataset.targets.values))
        z = tuple([y])
        result_dataset = Tuple(tup=z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("AdaBoostClassifierPredictor")
class AdaBoostClassifierPredictor(Task):
    """
    Predictor of a trained Adaboost classifier. This process predicts classes for a dataset.
    The predicted class of an input sample is computed as the weighted mean prediction of the classifiers in the ensemble.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': AdaBoostClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        abc = learned_model.get_result()
        y = abc.predict(dataset.features.values)
        result_dataset = Dataset(targets=DataFrame(y))
        return {'result' : result_dataset}