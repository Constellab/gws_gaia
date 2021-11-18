# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam,
                        StrParam, ScatterPlot2DView, ScatterPlot3DView, TableView, view, ResourceRField, FloatRField, IntRField)
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

@resource_decorator("LDAResult", 
                    human_name="LDA Result", 
                    short_description = "Linear Discriminant Analysis result", 
                    hide=True)
class LDAResult(BaseResource):

    _training_set: Resource = ResourceRField() #pour lier ressources entre elles
#    _log_likelihood: int = FloatRField() #list, float, dict,...
    _nb_components: int = IntRField()

    def _get_transformed_data(self) -> DataFrame: #retourne DataFrame
        lda: LinearDiscriminantAnalysis = self.get_result() #typage de lda du type LDA
        ncomp = self._nb_components
        X_transformed: DataFrame = lda.transform(self._training_set.get_features().values)
        columns = [f"PC{n+1}" for n in range(0,ncomp)]
        X_transformed = DataFrame(data=X_transformed, columns=columns, index=self._training_set.instance_names)
        return X_transformed

    # def _get_log_likelihood(self) -> float:
    #     if not self._log_likelihood:
    #         pca = self.get_result()
    #         self._log_likelihood = pca.score(self._training_set.get_features().values)
    #     return self._log_likelihood

    @view(view_type=TableView, human_name="ProjectedDataTable' table", short_description="Table of data in the score plot")
    def view_transformed_data_as_table(self, params: ConfigParams = None) -> dict:
        """
        View 2D score plot
        """
    
        x_transformed = self._get_transformed_data() 
        return TableView(data=x_transformed)

    @view(view_type=TableView, human_name="VarianceTable", short_description="Table of explained variances")
    def view_variance_as_table(self, params: ConfigParams = None) -> dict:
        """
        View table data
        """

        lda = self.get_result()
        ncomp = self._nb_components
        index = [f"PC{n+1}" for n in range(0,ncomp)]
        columns = ["ExplainedVariance"]
        data = DataFrame(lda.explained_variance_ratio_, index=index, columns=columns)
        return TableView(data=data)

    @view(view_type=ScatterPlot2DView, human_name='ScorePlot2D', short_description='2D score plot')
    def view_scores_as_2d_plot(self, params: ConfigParams = None) -> dict:
        """
        View 2D score plot
        """

        x_transformed = self._get_transformed_data()
        view_model = ScatterPlot2DView(data=x_transformed)
        return view_model

    @view(view_type=ScatterPlot3DView, human_name='ScorePlot3D', short_description='3D score plot')
    def view_scores_as_3d_plot(self, params: ConfigParams = None) -> dict:
        """
        View 3D score plot
        """

        x_transformed = self._get_transformed_data()
        view_model = ScatterPlot3DView(data=x_transformed)
        return view_model

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
        result._training_set = dataset
        result._nb_components = params['nb_components']
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
    output_specs = {'result' : Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lda = learned_model.result
        x = lda.transform(dataset.get_features().values)
        result_dataset = Dataset(features = DataFrame(data=x))
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
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lda = learned_model.result
        y = lda.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = DataFrame(data=y))
        return {'result': result_dataset}