# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from sklearn.decomposition import PCA

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.dataset import Dataset
from ..data.core import GenericResult
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("PCAResult", hide=True)
class PCAResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("PCATrainer")
class PCATrainer(Task):
    """
    Trainer of a Principal Component Analysis (PCA) model. Fit a PCA model with a training dataset.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : PCAResult}
    config_specs = {
        'nb_components':IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        pca = PCA(n_components=params["nb_components"])
        pca.fit(dataset.features.values)
        result = PCAResult.from_result(result=pca)
        return {'result': result}
        
#==============================================================================
#==============================================================================

@task_decorator("PCATransformer")
class PCATransformer(Task):
    """
    Transformer of a Principal Component Analysis (PCA) model. Apply dimensionality reduction to a dataset.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html for more details

    """
    input_specs = {'dataset' : Dataset, 'learned_model': PCAResult}
    output_specs = {'result' : GenericResult}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        pca = learned_model.binary_store['result']
        x = pca.transform(dataset.features.values)
        result = GenericResult.from_result(result=x)
        return {'result': result}