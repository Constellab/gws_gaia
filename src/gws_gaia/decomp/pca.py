# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from pandas import DataFrame
import numpy as np
from sklearn.decomposition import PCA

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, BarPlotView,
                        StrParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, FloatRField, Resource)
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("PCATrainerResult", hide=True)
class PCATrainerResult(BaseResource):

    _training_set: Resource = ResourceRField() #pour lier ressources entre elles
    _log_likelihood: int = FloatRField() #list, float, dict,...

    def _get_transformed_data(self) -> DataFrame: #retourne DataFrame
        pca: PCA = self.get_result() #typage de pca du type PCA
        ncomp = pca.n_components_
        X_transformed: DataFrame = pca.transform(self._training_set.get_features().values)
        columns = [f"PC{n+1}" for n in range(0,ncomp)]
        X_transformed = DataFrame(data=X_transformed, columns=columns, index=self._training_set.instance_names)
        return X_transformed

    def _get_log_likelihood(self) -> float:
        if not self._log_likelihood:
            pca = self.get_result()
            self._log_likelihood = pca.score(self._training_set.get_features().values)
        return self._log_likelihood

    @view(view_type=TableView, human_name="ProjectedDataTable table", short_description="Table of data in the score plot")
    def view_transformed_data_as_table(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """
    
        x_transformed = self._get_transformed_data() 
        return TableView(data=x_transformed)

    @view(view_type=TableView, human_name="VarianceTable", short_description="Table of explained variances")
    def view_variance_as_table(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        pca = self.get_result()
        index = [f"PC{n+1}" for n in range(0,pca.n_components_)]
        columns = ["ExplainedVariance"]
        data = DataFrame(pca.explained_variance_ratio_, index=index, columns=columns)
        return TableView(data=data)

    @view(view_type=BarPlotView, human_name="VarianceBarPlot", short_description="Barplot of explained variances")
    def view_variance_as_barplot(self, params: ConfigParams) -> dict:
        """
        View bar plot of explained variances
        """

        pca = self.get_result()
        explained_var = pca.explained_variance_ratio_[np.newaxis]
        columns = [f"PC{n+1}" for n in range(0,pca.n_components_)]
        index = ["ExplainedVariance"]
        data = DataFrame(explained_var, columns=columns, index=index)
        
        return BarPlotView(data=data)

    @view(view_type=ScatterPlot2DView, human_name='ScorePlot2D', short_description='2D score plot')
    def view_scores_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        x_transformed = self._get_transformed_data()
        view_model = ScatterPlot2DView(data=x_transformed)
        return view_model

    @view(view_type=ScatterPlot3DView, human_name='ScorePlot3D', short_description='3D score plot')
    def view_scores_as_3d_plot(self, params: ConfigParams) -> dict:
        """
        View 3D score plot
        """

        x_transformed = self._get_transformed_data()
        view_model = ScatterPlot3DView(data=x_transformed)
        return view_model

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
        'nb_components':IntParam(default_value=2, min_value=2)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        ncomp = params["nb_components"]
        pca = PCA(n_components=ncomp)
        pca.fit(dataset.get_features().values)
        result = PCATrainerResult(result = pca)
        result._training_set = dataset
        return {'result': result}
        
#==============================================================================
#==============================================================================

@task_decorator("PCATransformer")
class PCATransformer(Task):
    """
    Transformer of a Principal Component Analysis (PCA) model. Apply dimensionality reduction to a dataset.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html for more details

    """
    input_specs = {'dataset' : Dataset, 'learned_model': PCATrainerResult}
    output_specs = {'result' : Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        pca = learned_model.get_result()
        X_transformed = pca.transform(dataset.get_features().values)
        result = Dataset(features = X_transformed)
        return {'result': result}
