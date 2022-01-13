# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from numpy import ravel
from pandas import DataFrame, concat
from sklearn.linear_model import Lasso

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam,
                        view, TableView, ResourceRField, ScatterPlot2DView, ScatterPlot3DView, FloatRField, 
                        DataFrameRField, BadRequestException, Table, Dataset)
from gws_core import Dataset
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# LassoResult
#
# *****************************************************************************

@resource_decorator("LassoResult", hide=True)
class LassoResult(BaseResource):

    _training_set: Resource = ResourceRField()
    _R2: int = FloatRField()

    def _get_predicted_data(self) -> DataFrame:
        las: Lasso = self.get_result() #lir du type Linear Regression
        Y_predicted: DataFrame = las.predict(self._training_set.get_features().values)
        Y_predicted = DataFrame(
            data=Y_predicted, 
            index=self._training_set.row_names, 
            columns=self._training_set.target_names
        )
        return Y_predicted

    def _get_R2(self) -> float:
        if not self._R2:
            lir = self.get_result()
            self._R2 = las.score(X=self._training_set.get_features().values, y=self._training_set.get_targets().values)
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
# LassoTrainer
#
# *****************************************************************************

@task_decorator("LassoTrainer")
class LassoTrainer(Task):
    """
    Trainer of a lasso model. Fit a lasso model with a training dataset.

    See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.Lasso.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : LassoResult}
    config_specs = {
        'alpha': FloatParam(default_value=1, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        las = Lasso(alpha=params["alpha"])
        las.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = LassoResult(result = las)
        result._training_set = dataset                
        return {'result': result}

# *****************************************************************************
#
# LassoPredictor
#
# *****************************************************************************

@task_decorator("LassoPredictor")
class LassoPredictor(Task):
    """
    Predictor of a lasso model. Predict target values from a dataset with a trained lasso model.

    See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.Lasso.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LassoResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        las = learned_model.result
        y = las.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y, 
            row_names=dataset.row_names, 
            column_names=dataset.target_names, 
            target_names=dataset.target_names
        )
        return {'result': result_dataset}