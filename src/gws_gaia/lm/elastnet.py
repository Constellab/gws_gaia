# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (BadRequestException, ConfigParams, DataFrameRField,
                      Dataset, FloatParam, FloatRField, IntParam, Resource,
                      ResourceRField, ScatterPlot2DView, ScatterPlot3DView,
                      StrParam, Table, TabularView, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, view, 
                      InputSpec, OutputSpec)
from numpy import ravel
from pandas import DataFrame, concat
from sklearn.linear_model import ElasticNet

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# ElasticNetResult
#
# *****************************************************************************


@resource_decorator("ElasticNetResult", hide=True)
class ElasticNetResult(BaseResourceSet):
    """ ElasticNetResult """

    PREDICTION_TABLE_NAME = "Prediction table"
    _R2: int = FloatRField()

    def _get_predicted_data(self) -> DataFrame:
        eln: ElasticNet = self.get_result()  # lir du type Linear Regression
        Y_predicted: DataFrame = eln.predict(self._training_set.get_features().values)
        Y_predicted = DataFrame(
            data=Y_predicted,
            index=self._training_set.row_names,
            columns=self._training_set.target_names
        )
        return Y_predicted

    def _get_R2(self) -> float:
        if not self._R2:
            lir = self.get_result()
            self._R2 = eln.score(X=self._training_set.get_features().values, y=self._training_set.get_targets().values)
        return self._R2

    @view(view_type=TabularView, human_name="PredictionTable", short_description="Prediction Table")
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
# ElasticNetTrainer
#
# *****************************************************************************


@task_decorator("ElasticNetTrainer", human_name="ElasticNet trainer",
                short_description="Train an ElasticNet model")
class ElasticNetTrainer(Task):
    """
    Trainer of an elastic net model. Fit model with coordinate descent.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for more details
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(ElasticNetResult, human_name="result", short_description="The output result")}
    config_specs = {
        'alpha': FloatParam(default_value=1, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        eln = ElasticNet(alpha=params["alpha"])
        eln.fit(dataset.get_features().values, dataset.get_targets().values)
        result = ElasticNetResult(training_set=dataset, result=eln)
        return {'result': result}

# *****************************************************************************
#
# ElasticNetPredictor
#
# *****************************************************************************


@task_decorator("ElasticNetPredictor", human_name="Elastic-Net predictor",
                short_description="Predict dataset targets using a trained Elastic-Net model")
class ElasticNetPredictor(Task):
    """
    Predictor of a trained elastic net model. Predict from a dataset using the trained model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for more details
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(ElasticNetResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        eln = learned_model.get_result()
        y = eln.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
