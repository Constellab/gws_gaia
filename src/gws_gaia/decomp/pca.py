# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from sklearn.decomposition import PCA

from gws_core import (Task, Resource, task_decorator, resource_decorator)
from ..data.dataset import Dataset
from ..data.core import Tuple

#==============================================================================
#==============================================================================

@resource_decorator("PCAResult", hide=True)
class PCAResult(Resource):
    def __init__(self, pca: PCA = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_store['pca'] = pca

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
        'nb_components': {"type": 'int', "default": 2, "min": 0}
    }

    async def task(self):
        dataset = self.input['dataset']
        pca = PCA(n_components=self.get_param("nb_components"))
        pca.fit(dataset.features.values)

        t = self.output_specs["result"]
        result = t(pca=pca)
        self.output['result'] = result
        
#==============================================================================
#==============================================================================

@task_decorator("PCATransformer")
class PCATransformer(Task):
    """
    Transformer of a Principal Component Analysis (PCA) model. Apply dimensionality reduction to a dataset.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html for more details

    """
    input_specs = {'dataset' : Dataset, 'learned_model': PCAResult}
    output_specs = {'result' : Tuple}
    config_specs = {
    }

    async def task(self):
        dataset = self.input['dataset']
        learned_model = self.input['learned_model']
        pca = learned_model.kv_store['pca']
        x = pca.transform(dataset.features.values)

        t = self.output_specs["result"]
        result = t(tuple=x)
        #a = result.kv_store['pca']
        self.output['result'] = result