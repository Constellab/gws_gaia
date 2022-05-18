# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from gws_core import (BoolParam, ConfigParams, Dataset, FloatRField, IntParam,
                      Resource, ResourceRField, ScatterPlot2DView,
                      ScatterPlot3DView, Table, TabularView, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator, view, TechnicalInfo, 
                      InputSpec, OutputSpec)
from pandas import DataFrame, concat
import numpy as np
from sklearn.cross_decomposition import PLSRegression

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# PLSTrainerResult
#
# *****************************************************************************


@resource_decorator("PLSTrainerResult", hide=True)
class PLSTrainerResult(BaseResourceSet):
    """ PLSTrainerResult """
    TRANSFORMED_TABLE_NAME = "Transformed data table"
    VARIANCE_TABLE_NAME = "Variance table"
    PREDICTION_TABLE_NAME = "Prediction table"
    _r2: int = FloatRField()

    def __init__(self, training_set=None, result=None):
        super().__init__(training_set=training_set, result=result)
        # append tables
        if training_set is not None:
            self._create_transformed_table()
            self._create_prediction_table()
            self._get_r2()

    def _create_transformed_table(self):
        pls: PLSRegression = self.get_result()
        ncomp = pls.x_rotations_.shape[1]
        data: DataFrame = pls.transform(self.get_training_set().get_features().values)
        columns = [f"PC{i+1}" for i in range(0, ncomp)]
        data = DataFrame(data=data, columns=columns, index=self.get_training_set().row_names)
        table = Table(data=data)
        row_tags = self.get_training_set().get_row_tags()
        table.name = self.TRANSFORMED_TABLE_NAME
        table.set_row_tags(row_tags)
        self.add_resource(table)

    def _create_prediction_table(self) -> DataFrame:
        pls: PLSRegression = self.get_result()  # lir du type Linear Regression

        data: DataFrame = np.concatenate((self.get_training_set().get_targets().values, pls.predict(self.get_training_set().get_features().values)), axis=1)
        columns = ["measured_" + self.get_training_set().target_names[0], "predicted_" +  self.get_training_set().target_names[0]]
        data = DataFrame(
            data=data,
            columns=columns,
            index=self.get_training_set().row_names
        )

        table = Table(data=data)
        table.name = self.PREDICTION_TABLE_NAME
        row_tags = self.get_training_set().get_row_tags()
        table.set_row_tags(row_tags)
        self.add_resource(table)

    def get_transformed_table(self):
        """ Get transformed table """
        if self.resource_exists(self.TRANSFORMED_TABLE_NAME):
            return self.get_resource(self.TRANSFORMED_TABLE_NAME)
        else:
            return None

    def get_prediction_table(self):
        """ Get prediction table """
        if self.resource_exists(self.PREDICTION_TABLE_NAME):
            return self.get_resource(self.PREDICTION_TABLE_NAME)
        else:
            return None

    def _get_r2(self) -> float:
        """ Get R2 """
        if not self._r2:
            pls = self.get_result()
            self._r2 = pls.score(
                X=self.get_training_set().get_features().values,
                y=self.get_training_set().get_targets().values
            )
        technical_info = TechnicalInfo(key='R2', value=self._r2)
        self.add_technical_info(technical_info)

    @view(view_type=ScatterPlot2DView, human_name='2D-score plot', short_description='2D-score plot')
    def view_scores_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        data: DataFrame = self.get_transformed_table().get_data()
        _view = ScatterPlot2DView()
        row_tags = self.get_training_set().get_row_tags()
        _view.add_series(
            x=data['PC1'].to_list(),
            y=data['PC2'].to_list(),
            tags=row_tags
        )
        _view.x_label = 'PC1'
        _view.y_label = 'PC2'
        return _view

    @view(view_type=ScatterPlot2DView, human_name='Prediction plot', short_description='Prediction plot')
    def view_predictions_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View the target data and the predicted data in a 2d scatter plot. Works for data with only one target.
        """

        y_data = self.get_training_set().get_targets()
        y_predicted = self._get_predicted_data()
        row_tags = self.get_training_set().get_row_tags()
        _view = ScatterPlot2DView()
        for name in y_data.columns:
            _view.add_series(
                x=y_data.loc[:, name].values.tolist(),
                y=y_predicted.loc[:, name].values.tolist(),
                y_name=name,
                tags=row_tags
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
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(PLSTrainerResult, human_name="result", short_description="The output result")}
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
        result = PLSTrainerResult(training_set=dataset, result=pls)
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
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(PLSTrainerResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
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
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(PLSTrainerResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
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
