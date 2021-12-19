# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from gws_core import (BadRequestException, ConfigParams, DataFrameRField,
                      Dataset, FloatParam, FloatRField, IntParam, Resource,
                      ResourceRField, ScatterPlot2DView, ScatterPlot3DView,
                      StrParam, TableView, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, view)
from pandas import DataFrame, concat
from pandas.api.types import is_string_dtype
from sklearn.cross_decomposition import PLSRegression

from ..base.base_resource import BaseResource

# ==============================================================================
# ==============================================================================


@resource_decorator("PLSTrainerResult", hide=True)
class PLSTrainerResult(BaseResource):

    _training_set: Resource = ResourceRField()
    _R2: int = FloatRField()

    def _get_transformed_data(self):
        pls: PLSRegression = self.get_result()
        ncomp = pls.x_rotations_.shape[1]
        x_transformed: DataFrame = pls.transform(self._training_set.get_features().values)
        columns = [f"PC{n+1}" for n in range(0, ncomp)]
        x_transformed = DataFrame(data=x_transformed, columns=columns, index=self._training_set.instance_names)
        return x_transformed

    def _get_target_data(self) -> DataFrame:
        y_data: DataFrame = self._training_set.get_targets().values
        y_data = DataFrame(data=y_data)
        return y_data

    def _get_predicted_data(self) -> DataFrame:
        pls: PLSRegression = self.get_result()  # lir du type Linear Regression
        y_predicted: DataFrame = pls.predict(self._training_set.get_features().values)
        y_predicted = DataFrame(data=y_predicted)
        return y_predicted

    def _get_R2(self) -> float:
        if not self._R2:
            pls = self.get_result()
            self._R2 = pls.score(X=self._training_set.get_features().values, y=self._training_set.get_targets().values)
        return self._R2

    @view(view_type=TableView, human_name="ProjectedDataTable", short_description="Table of data in the score plot")
    def view_transformed_data_as_table(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        x_transformed = self._get_transformed_data()
        return TableView(data=x_transformed)

    @view(view_type=ScatterPlot2DView, human_name='ScorePlot2D', short_description='2D score plot')
    def view_scores_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        data: DataFrame = self._get_transformed_data()
        _view = ScatterPlot2DView()
        _view.add_series(
            x=data['PC1'].to_list(),
            y=data['PC2'].to_list()
        )
        _view.x_label = 'PC1'
        _view.y_label = 'PC2'
        return _view

    @view(view_type=ScatterPlot3DView, human_name='ScorePlot3D', short_description='3D score plot')
    def view_scores_as_3d_plot(self, params: ConfigParams) -> dict:
        """
        View 3D score plot
        """

        data: DataFrame = self._get_transformed_data()
        _view = ScatterPlot2DView()
        view.add_series(
            x=data['PC1'].to_list(),
            y=data['PC2'].to_list(),
            z=data['PC3'].to_list()
        )
        _view.x_label = 'PC1'
        _view.y_label = 'PC2'
        _view.z_label = 'PC3'
        return _view

    @view(view_type=TableView, human_name="PredictionTable", short_description="Prediction table")
    def view_predictions_as_table(self, params: ConfigParams) -> dict:
        """
        View the target data and the predicted data in a table. Works for data with only one target.
        """
        y_data = self._get_target_data()
        y_predicted = self._get_predicted_data()
        Y = concat([y_data, y_predicted], axis=1, ignore_index=True)
        data = Y.set_axis(["y_data", "y_predicted"], axis=1)
        return TableView(data=data)

    @view(view_type=ScatterPlot2DView, human_name='ScorePlot2D', short_description='2D data plot')
    def view_predictions_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View the target data and the predicted data in a 2d scatter plot. Works for data with only one target.
        """

        y_data = self._get_target_data()
        y_predicted = self._get_predicted_data()
        _view = ScatterPlot2DView()
        _view.add_series(
            x=y_data.loc[:, 0].values.tolist(),
            y=y_predicted.loc[:, 0].values.tolist()
        )
        _view.x_label = 'Y data'
        _view.y_label = 'Y predicted'
        return _view
# ==============================================================================
# ==============================================================================


@task_decorator("PLSTrainer")
class PLSTrainer(Task):
    """
    Trainer of a Partial Least Squares (PLS) regression model. Fit a PLS regression model to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details.
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': PLSTrainerResult}
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

# ==============================================================================
# ==============================================================================


@task_decorator("PLSTransformer")
class PLSTransformer(Task):
    """
    Learn and apply the dimension reduction on the train data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details
    """
    input_specs = {'dataset': Dataset, 'learned_model': PLSTrainerResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        pls = learned_model.get_result()
        x_transformed = pls.transform(dataset.get_features().values)
        result_dataset = Dataset(features=x_transformed)
        return {'result': result_dataset}

# ==============================================================================
# ==============================================================================


@task_decorator("PLSPredictor")
class PLSPredictor(Task):
    """
    Predictor of a Partial Least Squares (PLS) regression model. Predict targets of a dataset with a trained PLS regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details.
    """
    input_specs = {'dataset': Dataset, 'learned_model': PLSTrainerResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        pls = learned_model.get_result()
        Y = pls.predict(dataset.get_features().values)
        result_dataset = Dataset(targets=Y)
        return {'result': result_dataset}
