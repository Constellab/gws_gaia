# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (BadRequestException, ConfigParams, DataFrameRField,
                      Dataset, FloatParam, FloatRField, IntParam, Resource,
                      ResourceRField, ScatterPlot2DView, ScatterPlot3DView,
                      StrParam, Table, TabularView, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, view)
from numpy import ravel
from pandas import DataFrame, concat
from sklearn.linear_model import LinearRegression

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
        lir: LinearRegression = self.get_result()  # lir du type Linear Regression
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

    @view(view_type=TabularView, human_name="Prediction table")
    def view_predictions_as_table(self, params: ConfigParams) -> dict:
        """
        View the target data and the predicted data in a table. Works for data with only one target
        """
        Y_data = self._training_set.get_targets()
        Y_predicted = self._get_predicted_data()
        Y = concat([Y_data, Y_predicted], axis=1)
        data = Y.set_axis(["YData", "YPredicted"], axis=1)
        t_view = TabularView()
        t_view.set_data(data=data)
        return t_view

    @view(view_type=ScatterPlot2DView, human_name='2D-score plot')
    def view_predictions_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View the target data and the predicted data in a 2d scatter plot. Works for data with only one target
        """

        y_data = self._training_set.get_targets()
        y_predicted = self._get_predicted_data()
        _view = ScatterPlot2DView()
        for name in y_data.columns:
            _view.add_series(
                x=y_data.loc[:, name].values.tolist(),
                y=y_predicted.loc[:, name].values.tolist(),
                y_name=name
            )
        _view.x_label = 'YData'
        _view.y_label = 'YPredicted'
        return _view

# *****************************************************************************
#
# LinearRegressionTrainer
#
# *****************************************************************************


@task_decorator("LinearRegressionTrainer", human_name="Linear regression trainer",
                short_description="Train a linear regression model")
class LinearRegressionTrainer(Task):
    """
    Trainer fo a linear regression model. Fit a linear regression model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details.
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': LinearRegressionResult}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        lir = LinearRegression()
        lir.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = LinearRegressionResult(training_set=dataset, result=lir)
        return {'result': result}

# *****************************************************************************
#
# LinearRegressionPredictor
#
# *****************************************************************************


@task_decorator("LinearRegressionPredictor", human_name="Linear regression predictor",
                short_description="Predict dataset targets using a trained linear regression model")
class LinearRegressionPredictor(Task):
    """
    Predictor of a linear regression model. Predict target values of a dataset with a trained linear regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html for more details.
    """
    input_specs = {'dataset': Dataset, 'learned_model': LinearRegressionResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lir = learned_model.get_result()
        y = lir.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=DataFrame(y),
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
