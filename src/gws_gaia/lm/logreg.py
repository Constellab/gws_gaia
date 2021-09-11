# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

from gws_core import (Task, Resource, task_decorator, resource_decorator)

from ..data.core import Tuple
from ..data.dataset import Dataset

#==============================================================================
#==============================================================================

@resource_decorator("LogisticRegressionResult", hide=True)
class LogisticRegressionResult(Resource):
    def __init__(self, logreg: LogisticRegression = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['logreg'] = logreg

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
        'inv_reg_strength': {"type": 'float', "default": 1, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        logreg = LogisticRegression(C=self.get_param("inv_reg_strength"))
        logreg.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(logreg=logreg)
        self.output['result'] = result

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
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        logreg = learned_model.kv_store['logreg']
        y = logreg.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        
        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

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
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        logreg = learned_model.kv_store['logreg']
        y = logreg.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset