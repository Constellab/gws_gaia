# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.model import Process, Config, Resource

from sklearn.tree import DecisionTreeClassifier
from gaia.data import Tuple
from numpy import ravel

#========================================================================================
#========================================================================================

class Result(Resource):
    def __init__(self, dtc: DecisionTreeClassifier = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['dtc'] = dtc

#========================================================================================
#========================================================================================

class Trainer(Process):
    """ Trainer of the decision tree classifier. Build a decision tree classifier from the training set. 
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'max_depth': {"type": 'int', "default": None, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        dtc = DecisionTreeClassifier(max_depth=self.get_param("max_depth"))
        dtc.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(dtc=dtc)
        self.output['result'] = result

#========================================================================================
#========================================================================================

class Tester(Process):
    """
    Tester of a trained decision tree classifier. Return the mean accuracy on a given test data and labels for a trained decision tree classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        dtc = learned_model.kv_store['dtc']
        y = dtc.score(dataset.features.values,ravel(dataset.targets.values))
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#========================================================================================
#========================================================================================

class Predictor(Process):
    """ Predictor of a trained decision tree classifier. Predict class or regression value for a dataset. For a classification model, the predicted class for each sample in the dataset is returned. For a regression model, the predicted value based on the dataset is returned.

    See https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        dtc = learned_model.kv_store['dtc']
        y = dtc.predict(dataset.features.values)

        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset