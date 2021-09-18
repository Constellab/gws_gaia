# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#========================================================================================
#========================================================================================

@resource_decorator("DecisionTreeClassifierResult")
class DecisionTreeClassifierResult(BaseResource):
    pass

#========================================================================================
#========================================================================================

@task_decorator("DecisionTreeClassifierTrainer")
class DecisionTreeClassifierTrainer(Task):
    """ Trainer of the decision tree classifier. Build a decision tree classifier from the training set. 
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : DecisionTreeClassifierResult}
    config_specs = {
        'max_depth':IntParam(default_value=None, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        dtc = DecisionTreeClassifier(max_depth=params["max_depth"])
        dtc.fit(dataset.features.values, ravel(dataset.targets.values))
        result = DecisionTreeClassifierResult.from_result(result=dtc)
        return {'result': result}

#========================================================================================
#========================================================================================

@task_decorator("DecisionTreeClassifierTester")
class DecisionTreeClassifierTester(Task):
    """
    Tester of a trained decision tree classifier. Return the mean accuracy on a given test data and labels for a trained decision tree classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': DecisionTreeClassifierResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        dtc = learned_model.binary_store['result']
        y = dtc.score(dataset.features.values,ravel(dataset.targets.values))
        z = tuple([y])
        result_dataset = GenericResult.from_result(result = z)
        return {'result': result_dataset}

#========================================================================================
#========================================================================================

@task_decorator("DecisionTreeClassifierPredictor")
class DecisionTreeClassifierPredictor(Task):
    """ Predictor of a trained decision tree classifier. Predict class or regression value for a dataset. For a classification model, the predicted class for each sample in the dataset is returned. For a regression model, the predicted value based on the dataset is returned.

    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': DecisionTreeClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        dtc = learned_model.binary_store['result']
        y = dtc.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}