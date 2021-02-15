# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gaia.dataset import Dataset
from pandas import DataFrame

from gws.model import Config
from gws.controller import Controller
from gws.model import Process, Config, Resource

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from gaia.data import Tuple
from numpy import ravel

class Result(Resource):
    def __init__(self, lda: LinearDiscriminantAnalysis = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['lda'] = lda

#==============================================================================
#==============================================================================

class Trainer(Process):
    """
    Trainer of a linear discriminant analysis classifier. Fit Linear Discriminant Analysis model according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : Result}
    config_specs = {
        'solver': {"type": 'str', "default": 'svd'},
        'nb_components': {"type": 'int', "default": None, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        lda = LinearDiscriminantAnalysis(solver=self.get_param("solver"),n_components=self.get_param("nb_components"))
        lda.fit(dataset.features.values, ravel(dataset.targets.values))
        
        t = self.output_specs["result"]
        result = t(lda=lda)
        self.output['result'] = result
        
#==============================================================================
#==============================================================================

class Transformer(Process):
    """
    Transformer of a linear discriminant analysis classifier. Project data to maximize class separation.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details

    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        lda = learned_model.kv_store['lda']
        x = lda.transform(dataset.features.values)

        t = self.output_specs["result"]
        result = t(tuple=x)
        #a = result.kv_store['pca']
        self.output['result'] = result
        
#==============================================================================
#==============================================================================

class Tester(Process):
    """
    Tester of a trained linear discriminant analysis classifier. Return the mean accuracy on a given dataset for a trained linear discriminant analysis classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Tuple}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        lda = learned_model.kv_store['lda']
        y = lda.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])

        t = self.output_specs["result"]
        result_dataset = t(tuple = z)
        self.output['result'] = result_dataset

#==============================================================================
#==============================================================================

class Predictor(Process):
    """
    Predictor of a linear discriminant analysis classifier. Predict class labels for samples in a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': Result}
    output_specs = {'result' : Dataset}
    config_specs = {   
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        lda = learned_model.kv_store['lda']
        y = lda.predict(dataset.features.values)
        
        t = self.output_specs["result"]
        result_dataset = t(targets = DataFrame(y))
        self.output['result'] = result_dataset