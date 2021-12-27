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
                        DataFrameRField, BadRequestException, Table, Dataset)
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# LinearRegressionResult
#
# *****************************************************************************

@resource_decorator("LinearRegressionResult", hide=True)
class LinearRegressionResult(BaseResource):

    _training_set: Resource = ResourceRField()
    _R2: int = FloatRField()

    def _get_predicted_data(self) -> DataFrame:
        lir: LinearRegression = self.get_result() #lir du type Linear Regression
        Y_predicted: DataFrame = lir.predict(self._training_set.get_features().values)
        Y_predicted = DataFrame(
            data=Y_predicted, 
            index=self._training_set.row_names, 
            columns=self._training_set.target_names
        )
        return Y_predicted

    def _get_R2(self) -> float:
        if not self._R2:
            lir = self.get_result()
            self._R2 = lir.score(X=self._training_set.get_features().values, y=self._training_set.get_targets().values)
        return self._R2

    @view(view_type=TableView, human_name="PredictionTable", short_description="Prediction Table")
    def view_predictions_as_table(self, params: ConfigParams) -> dict:
        """
        View the target data and the predicted data in a table. Works for data with only one target
        """
        Y_data = self._training_set.get_targets()
        Y_predicted = self._get_predicted_data()
        Y = concat([Y_data, Y_predicted],axis=1)
        data = Y.set_axis(["YData", "YPredicted"], axis=1)
        table = Table(data=data)
        return TableView(table)

    @view(view_type=ScatterPlot2DView, human_name='ScorePlot2D', short_description='2D data plot')
    def view_predictions_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View the target data and the predicted data in a 2d scatter plot. Works for data with only one target
        """

        Y_data = self._training_set.get_targets()
        Y_predicted = self._get_predicted_data()
        Y = concat([Y_data, Y_predicted],axis=1)
        data = Y.set_axis(["YData", "YPredicted"], axis=1)
        table = Table(data=data)
        view_model = ScatterPlot2DView(table)
        return view_model

# *****************************************************************************
#
# LinearRegressionTrainer
#
# *****************************************************************************

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

# *****************************************************************************
#
# LinearRegressionPredictor
#
# *****************************************************************************

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
        result_dataset = Dataset(
            data = DataFrame(y),
            row_names = dataset.row_names,
            column_names = dataset.target_names,
            target_names = dataset.target_names
        )
        return {'result': result_dataset}