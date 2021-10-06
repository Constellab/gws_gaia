# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from pandas import DataFrame
from sklearn.decomposition import PCA

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, 
                        StrParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view)
from ..data.dataset import Dataset
from ..data.core import GenericResult
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("PCATrainerResult", hide=True)
class PCATrainerResult(BaseResource):

    @view(view_type=TableView, human_name="Scores' table", short_description="Scores' table")
    def view_scores_as_table(self, **kwargs) -> dict:
        """
        View 2D score plot
        """

        scores = self.get_result()["scores"]
        return TableView(data=scores, **kwargs)

    @view(view_type=ScatterPlot2DView, human_name='2DScorePlot', short_description='2D score plot')
    def view_scores_as_2d_plot(self, **kwargs) -> dict:
        """
        View 2D score plot
        """

        scores = self.get_result()["scores"]
        return ScatterPlot2DView(data=scores, x_column_name="PC1", y_column_names=["PC2"], **kwargs)

    @view(view_type=ScatterPlot3DView, human_name='3DScorePlot', short_description='3D score plot')
    def view_scores_as_3d_plot(self, **kwargs) -> dict:
        """
        View 3D score plot
        """

        scores = self.get_result()["scores"]
        return ScatterPlot3DView(data=scores, x_column_name="PC1", y_column_name="PC2", z_column_names=["PC3"], **kwargs)

#==============================================================================
#==============================================================================

@task_decorator("PCATrainer")
class PCATrainer(Task):
    """
    Trainer of a Principal Component Analysis (PCA) model. Fit a PCA model with a training dataset.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : PCATrainerResult}
    config_specs = {
        'nb_components':IntParam(default_value=2, min_value=1)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        ncomp = params["nb_components"]
        pca = PCA(n_components=ncomp)
        pca.fit(dataset.features.values)
        
        scores: DataFrame = pca.transform(dataset.features.values)
        columns = [f"PC{n+1}" for n in range(0,ncomp)]
        scores = DataFrame(data=scores, columns=columns, index=dataset.instance_names)

        result = PCATrainerResult(result = {
            "pca": pca,
            "scores": scores
        })
        return {'result': result}
        
#==============================================================================
#==============================================================================

@resource_decorator("PCATransformerResult", hide=True)
class PCATransformerResult(BaseResource):
    pass

@task_decorator("PCATransformer")
class PCATransformer(Task):
    """
    Transformer of a Principal Component Analysis (PCA) model. Apply dimensionality reduction to a dataset.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html for more details

    """
    input_specs = {'dataset' : Dataset, 'learned_model': PCATrainerResult}
    output_specs = {'result' : PCATransformerResult}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        pca = learned_model.get_result()["pca"]
        X_transformed = pca.transform(dataset.features.values)
        result = PCATransformerResult(result = X_transformed)
        return {'result': result}