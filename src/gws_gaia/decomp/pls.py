# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from gws_core import (BoolParam, ConfigParams, Dataset, FloatRField, IntParam,
                      Resource, ResourceRField, ScatterPlot2DView,
                      ScatterPlot3DView, TabularView, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, view)
from pandas import DataFrame, concat
from sklearn.cross_decomposition import PLSRegression

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# PLSTrainerResult
#
# *****************************************************************************


@resource_decorator("PLSTrainerResult", hide=True)
class PLSTrainerResult(BaseResource):

    _training_set: Resource = ResourceRField()
    _R2: int = FloatRField()

    def _get_transformed_data(self):
        pls: PLSRegression = self.get_result()
        ncomp = pls.x_rotations_.shape[1]
        x_transformed: DataFrame = pls.transform(self._training_set.get_features().values)
        columns = [f"PC{i+1}" for i in range(0, ncomp)]
        x_transformed = DataFrame(
            data=x_transformed,
            columns=columns,
            index=self._training_set.row_names
        )
        return x_transformed

    def _get_predicted_data(self) -> DataFrame:
        pls: PLSRegression = self.get_result()  # lir du type Linear Regression
        y_predicted: DataFrame = pls.predict(self._training_set.get_features().values)
        y_predicted = DataFrame(
            data=y_predicted,
            columns=self._training_set.target_names,
            index=self._training_set.row_names
        )
        return y_predicted

    def _get_R2(self) -> float:
        if not self._R2:
            pls = self.get_result()
            self._R2 = pls.score(
                X=self._training_set.get_features().values,
                y=self._training_set.get_targets().values
            )
        return self._R2

    @view(view_type=TabularView, human_name="Projected data table", short_description="Table of data in the score plot")
    def view_transformed_data_as_table(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        x_transformed = self._get_transformed_data()
        t_view = TabularView()
        t_view.set_data(data=x_transformed)
        return t_view

    @view(view_type=ScatterPlot2DView, default_view=True, human_name='2D-score plot', short_description='2D-score plot',
          specs={
              'show_labels':
              BoolParam(
                  default_value=False, human_name="Show labels",
                  short_description="Set True to see sample labels if they are provided; False otherwise")})
    def view_scores_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        data: DataFrame = self._get_transformed_data()
        _view = ScatterPlot2DView()
        targets = self._training_set.get_targets()
        show_labels = params.get('show_labels')
        targets = self._training_set.get_targets()
        if targets.shape[1] == 1:
            if self._training_set.has_string_targets():
                show_labels = True
        else:
            show_labels = False

        if show_labels:
            labels = sorted(list(set(targets.transpose().values.tolist()[0])))
            for lbl in labels:
                idx = (targets == lbl)
                _view.add_series(
                    x=data[idx, 'PC1'].to_list(),
                    y=data[idx, 'PC2'].to_list(),
                    y_name=lbl
                )
        else:
            _view.add_series(
                x=data['PC1'].to_list(),
                y=data['PC2'].to_list()
            )
        _view.x_label = 'PC1'
        _view.y_label = 'PC2'
        return _view

    # @view(view_type=ScatterPlot3DView, human_name='3D-score plot', short_description='3D-score plot')
    # def view_scores_as_3d_plot(self, params: ConfigParams) -> dict:
    #     """
    #     View 3D score plot
    #     """

    #     data: DataFrame = self._get_transformed_data()
    #     _view = ScatterPlot2DView()
    #     view.add_series(
    #         x=data['PC1'].to_list(),
    #         y=data['PC2'].to_list(),
    #         z=data['PC3'].to_list()
    #     )
    #     _view.x_label = 'PC1'
    #     _view.y_label = 'PC2'
    #     _view.z_label = 'PC3'
    #     return _view

    @view(view_type=TabularView, human_name="Prediction table", short_description="Prediction table")
    def view_predictions_as_table(self, params: ConfigParams) -> dict:
        """
        View the target data and the predicted data in a table. Works for data with only one target.
        """
        y_data = self._training_set.get_targets()
        y_predicted = self._get_predicted_data()
        Y = concat([y_data, y_predicted], axis=1)
        data = Y.set_axis(["YData", "YPredicted"], axis=1)
        t_view = TabularView()
        t_view.set_data(data=data)
        return t_view

    @view(view_type=ScatterPlot2DView, human_name='Prediction plot', short_description='Prediction plot')
    def view_predictions_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View the target data and the predicted data in a 2d scatter plot. Works for data with only one target.
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
# PLSTrainer
#
# *****************************************************************************


@task_decorator("PLSTrainer", human_name="PLS trainer",
                short_description="Train a Partial Least Squares (PLS) regression model")
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

# *****************************************************************************
#
# PLSTransformer
#
# *****************************************************************************


@task_decorator("PLSTransformer", human_name="PLS transformer",
                short_description="Apply the PLS dimension reduction on a training dataset")
class PLSTransformer(Task):
    """
    Transformer of a Partial Least Squares (PLS) regression model. Apply the dimensionality reduction to a dataset.

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
        ncomp = x_transformed.shape[1]
        result_dataset = Dataset(
            data=x_transformed,
            row_names=dataset.row_names,
            column_names=["PC"+str(i+1) for i in range(0, ncomp)]
        )
        return {'result': result_dataset}

# *****************************************************************************
#
# PLSPredictor
#
# *****************************************************************************


@task_decorator("PLSPredictor", human_name="PLS predictor",
                short_description="Predict dataset targets using a trained PLS regression model")
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
        result_dataset = Dataset(
            data=Y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
