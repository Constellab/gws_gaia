# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

@resource_decorator("LDAResult", 
                    human_name="LDA Result", 
                    short_description = "Linear Discriminant Analysis result", 
                    hide=True)

class LDAResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("LDATrainer")
class LDATrainer(Task):
    """
    Trainer of a linear discriminant analysis classifier. Fit Linear Discriminant Analysis model according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : LDAResult}
    config_specs = {
        'solver': StrParam(default_value='svd'),
        'nb_components': IntParam(default_value=None, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        lda = LinearDiscriminantAnalysis(solver=params["solver"],n_components=params["nb_components"])
        lda.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = LDAResult(result = lda)
        return {'result': result}
        
#==============================================================================
#==============================================================================

@task_decorator("LDATransformer")
class LDATransformer(Task):
    """
    Transformer of a linear discriminant analysis classifier. Project data to maximize class separation.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details

    """
    input_specs = {'dataset' : Dataset, 'learned_model': LDAResult}
    output_specs = {'result' : GenericResult}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lda = learned_model.result
        x = lda.transform(dataset.get_features().values)
        result = GenericResult(result = x)
        return {'result': result}
        
#==============================================================================
#==============================================================================

@task_decorator("LDATester")
class LDATester(Task):
    """
    Tester of a trained linear discriminant analysis classifier. Return the mean accuracy on a given dataset for a trained linear discriminant analysis classifier.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LDAResult}
    output_specs = {'result' : GenericResult}
    config_specs = { }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lda = learned_model.result
        y = lda.score(dataset.get_features().values, dataset.get_targets().values)
        z = tuple([y])
        result_dataset = GenericResult(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("LDAPredictor")
class LDAPredictor(Task):
    """
    Predictor of a linear discriminant analysis classifier. Predict class labels for samples in a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LDAResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lda = learned_model.result
        y = lda.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}