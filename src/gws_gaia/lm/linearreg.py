# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame, concat
from sklearn.linear_model import LinearRegression

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam,
                        view, TableView, ResourceRField, ScatterPlot2DView, ScatterPlot3DView, FloatRField, 
                        DataFrameRField, BadRequestException)

from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("LinearRegressionResult", hide=True)
class LinearRegressionResult(BaseResource):

    _training_set: Resource = ResourceRField()
    _R2: int = FloatRField()

    def _get_target_data(self) -> DataFrame:
        Y_data: DataFrame = self._training_set.get_targets().values
        Y_data = DataFrame(data=Y_data)
        return Y_data

    def _get_predicted_data(self) -> DataFrame:
        lir: LinearRegression = self.get_result() #lir du type Linear Regression
        Y_predicted: DataFrame = lir.predict(self._training_set.get_features().values)
        Y_predicted = DataFrame(data=Y_predicted)
        return Y_predicted

    def _get_R2(self) -> float:
        if not self._R2:
            lir = self.get_result()
            self._R2 = lir.score(X=self._training_set.get_features().values, y=self._training_set.get_targets().values)
        return self._R2

    @view(view_type=TableView, human_name="Table", short_description="Table")
    def view_predictions_as_table(self, *args, **kwargs) -> dict:
        """
        View the target data and the predicted data in a table
        """
        Y_data = self._get_target_data()
        Y_predicted = self._get_predicted_data()
        Y = concat([Y_data, Y_predicted],axis=1, ignore_index=True)
        data = Y.set_axis(["Y_data", "Y_predicted"], axis=1)
        # data = DataFrame(data=Y, columns=columns)

        return TableView(
            data=data, 
            #title="Target data and predicted data", 
            *args, **kwargs
        )

    @view(view_type=ScatterPlot2DView, human_name='ScorePlot2D', short_description='2D data plot')
    def view_predictions_as_2d_plot(self, *args, **kwargs) -> dict:
        """
        View the target data and the predicted data in a 2d scatter plot
        """

        Y_data = self._get_target_data()
        Y_predicted = self._get_predicted_data()
        Y = concat([Y_data, Y_predicted],axis=1, ignore_index=True)
        data = Y.set_axis(["Y_data", "Y_predicted"], axis=1)
        #data = DataFrame(data=Y, columns=columns)

        view_model = ScatterPlot2DView(
            data=data, #prend DataFrame, Table, Dataset
            #title="Predicted data versus target data", 
            #subtitle="R2 = {:.2f}".format(self._get_R2()), 
            *args, **kwargs
        )
        return view_model

#==============================================================================
#==============================================================================

# @resource_decorator("LinearRegressionResult", hide=True)
# class LinearRegressionPredictorResult(BaseResource):

#     _training_set: Resource = ResourceRField()

#     def _get_data(self) -> DataFrame:
#         data: DataFrame = self._training_set.get_features().values
#         data = DataFrame(data=data, index=self._training_set.instance_names)
#         return data

#     @view(view_type=ScatterPlot2DView, human_name='ScorePlot3D', short_description='2D score plot')
#     def view_prediction_as_2d_plot(self, *args, **kwargs) -> dict:
#         """
#         View 2D score plot
#         """

#         x = [self._data_set, self._prediction_target]
#         view_model = ScatterPlot2DView(
#             data=x, 
#             #title="Transformed data", 
#             #subtitle="log-likelihood = {:.2f}".format(self._get_log_likelihood()), 
#             *args, **kwargs
#         )
#         return view_model


#==============================================================================
#==============================================================================

@task_decorator("LinearRegressionTrainer")
class LinearRegressionTrainer(Task):
    """
    Trainer fo a linear regression model. Fit a linear regression model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : LinearRegressionResult}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        lir = LinearRegression()
        lir.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = LinearRegressionResult(result = lir)
        result._training_set = dataset
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("LinearRegressionTester")
class LinearRegressionTester(Task):
    """
    Tester of a trained linear regression model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained linear regression model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LinearRegressionResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lir = learned_model.result
        y = lir.score(dataset.get_features().values, dataset.get_targets().values)
        z = tuple([y])
        result_dataset = GenericResult(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("LinearRegressionPredictor")
class LinearRegressionPredictor(Task):
    """
    Predictor of a linear regression model. Predict target values of a dataset with a trained linear regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LinearRegressionResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lir = learned_model.result
        y = lir.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}