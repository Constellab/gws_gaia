# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from sklearn.ensemble import AdaBoostClassifier
from pandas import DataFrame

from gws_core import (Task, Resource, task_decorator, resource_decorator)
from ..data.core import Tuple
from ..data.dataset import Dataset

@resource_decorator("AdaBoostClassifierResult", hide=True)
class AdaBoostClassifierResult(Resource):
    def __init__(self, *args, abc: AdaBoostClassifier = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['abc'] = abc

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
        'nb_estimators': {"type": 'int', "default": 50, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        abc = AdaBoostClassifier(n_estimators=self.get_param("nb_estimators"))
        abc.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(abc=abc)
        self.output['result'] = result

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
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        abc = learned_model.kv_store['abc']
        y = abc.score(dataset.features.values,ravel(dataset.targets.values))
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

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
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        abc = learned_model.kv_store['abc']
        y = abc.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset