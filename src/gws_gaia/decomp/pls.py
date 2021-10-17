# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from pandas import DataFrame
from pandas.api.types import is_string_dtype
from sklearn.cross_decomposition import PLSRegression

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, 
                        FloatParam, StrParam, view, TableView, ScatterPlot2DView,
                        ScatterPlot3DView, ResourceRField, Resource, FloatRField, 
                        DataFrameRField, BadRequestException)
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("PLSTrainerResult", hide=True)
class PLSTrainerResult(BaseResource):

    _training_set: Resource = ResourceRField()
    _R2: int = FloatRField()

    def _get_transformed_data(self):
        pls: PLSRegression = self.get_result()
        ncomp = pls.x_rotations_.shape[1]
        X_transformed: DataFrame = pls.transform(self._training_set.get_features().values)
        columns = [f"PC{n+1}" for n in range(0,ncomp)]
        X_transformed = DataFrame(data=X_transformed, columns=columns, index=self._training_set.instance_names)
        return X_transformed
        
    def _get_R2(self) -> float:
        if not self._R2:
            pls = self.get_result()
            self._R2 = pls.score(X=self._training_set.get_features().values, y=self._training_set.get_targets().values)
        return self._R2


    @view(view_type=TableView, human_name="TransformedDataTable' table", short_description="Table of data in the score plot")
    def view_transformed_data_as_table(self, *args, **kwargs) -> dict:
        """
        View 2D score plot
        """

        x_transformed = self._get_transformed_data()
        return TableView(
            data=x_transformed, 
            title="Transformed data", 
            subtitle="R2 = {:.2f}".format(self._get_R2()), 
            *args, **kwargs
        )

    # @view(view_type=TableView, human_name="VarianceTable", short_description="Table of explained variances")
    # def view_variance_as_table(self, *args, **kwargs) -> dict:
    #     """
    #     View 2D score plot
    #     """

    #     pls = self.get_result()
    #     ncomp = pls.x_rotations_.shape[1]
    #     index = [f"PC{n+1}" for n in range(0,ncomp)]
    #     columns = ["ExplainedVariance"]
    #     data = DataFrame(pls.explained_variance_ratio_, index=index, columns=columns)
    #     return TableView(data=data, *args, **kwargs)

    @view(view_type=ScatterPlot2DView, human_name='ScorePlot3D', short_description='2D score plot')
    def view_scores_as_2d_plot(self, *args, **kwargs) -> dict:
        """
        View 2D score plot
        """

        x_transformed = self._get_transformed_data()
        view_model = ScatterPlot2DView(
            data=x_transformed, 
            title="Transformed data", 
            subtitle="R2 = {:.2f}".format(self._get_R2()), 
            *args, **kwargs
        )
        return view_model

    @view(view_type=ScatterPlot3DView, human_name='ScorePlot3D', short_description='3D score plot')
    def view_scores_as_3d_plot(self, *args, **kwargs) -> dict:
        """
        View 3D score plot
        """

        x_transformed = self._get_transformed_data()
        view_model = ScatterPlot3DView(
            data=x_transformed,
            title="Transformed data", 
            subtitle="R2 = {:.2f}".format(self._get_R2()), 
            *args, **kwargs
        )
        return view_model


#==============================================================================
#==============================================================================

@task_decorator("PLSTrainer")
class PLSTrainer(Task):
    """
    Trainer of a Partial Least Squares (PLS) regression model. Fit a PLS regression model to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : PLSTrainerResult}
    config_specs = {
        'nb_components': IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        ncomp = params["nb_components"]
        pls = PLSRegression(n_components=ncomp)
        if dataset.has_string_targets():
            y = self.convert_targets_to_dummy_matrix().values
        else:
            y = dataset.get_targets().values
        pls.fit(dataset.get_features().values, y)
        result = PLSTrainerResult(result=pls)
        result._training_set = dataset
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("PLSTransformer")
class PLSTransformer(Task):
    """
    Learn and apply the dimension reduction on the train data.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': PLSTrainerResult}
    output_specs = {'result' : Dataset}
    config_specs = {  }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        pls = learned_model.get_result()
        X_transformed = pls.transform(dataset.get_features().values)
        result_dataset = Dataset(features = X_transformed)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("PLSPredictor")
class PLSPredictor(Task):
    """
    Predictor of a Partial Least Squares (PLS) regression model. Predict targets of a dataset with a trained PLS regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': PLSTrainerResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        pls = learned_model.get_result()
        Y = pls.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = Y)
        return {'result': result_dataset}