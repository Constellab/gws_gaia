# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from typing import List

import numpy as np
import pandas
from gws_core import (BoolParam, ConfigParams, Dataset, FloatRField, InputSpec,
                      IntParam, OutputSpec, Resource, ResourceRField,
                      ScatterPlot2DView, ScatterPlot3DView, Table, TabularView,
                      Task, TaskInputs, TaskOutputs, TechnicalInfo,
                      resource_decorator, task_decorator, view)
from pandas import DataFrame, concat
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

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
            self._create_r2()
            self._create_variance_table()

    def _create_transformed_table(self):
        pls: PLSRegression = self.get_result()
        ncomp = pls.x_rotations_.shape[1]
        data: DataFrame = pls.transform(self.get_training_set().get_features().values)
        columns = [f"PC{i+1}" for i in range(0, ncomp)]
        data = DataFrame(data=data, columns=columns, index=self.get_training_set().row_names)
        table = Table(data=data)
        row_tags = self.get_training_set().get_row_tags()
        table.name = self.TRANSFORMED_TABLE_NAME
        table.set_all_rows_tags(row_tags)
        self.add_resource(table)

    def _create_prediction_table(self) -> DataFrame:
        pls: PLSRegression = self.get_result()  # lir du type Linear Regression
        training_set = self.get_training_set()
        if training_set.has_string_targets():
            y_true = training_set.convert_targets_to_dummy_matrix()
        else:
            y_true = training_set.get_targets()

        y_pred = pls.predict(training_set.get_features().values)

        columns = [
            *["true_" + col for col in y_true.columns],
            *["predicted_" + col for col in y_true.columns]
        ]

        data: DataFrame = np.concatenate((y_true.values, y_pred), axis=1)
        data = DataFrame(
            data=data,
            columns=columns,
            index=self.get_training_set().row_names
        )
        table = Table(data=data)
        table.name = self.PREDICTION_TABLE_NAME
        row_tags = self.get_training_set().get_row_tags()
        table.set_all_rows_tags(row_tags)
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

    def get_variance_table(self):
        """ Get variance table """

        if self.resource_exists(self.VARIANCE_TABLE_NAME):
            return self.get_resource(self.VARIANCE_TABLE_NAME)
        else:
            return None

    def _create_r2(self) -> float:
        """ Get R2 """
        if not self._r2:
            pls = self.get_result()
            training_set = self.get_training_set()
            if training_set.has_string_targets():
                y_true = training_set.convert_targets_to_dummy_matrix()
            else:
                y_true = training_set.get_targets()

            self._r2 = pls.score(
                X=training_set.get_features().values,
                y=y_true
            )
        technical_info = TechnicalInfo(key='R2', value=self._r2)
        self.add_technical_info(technical_info)

    def _create_variance_table(self) -> List[float]:
        training_set = self.get_training_set()
        if training_set.has_string_targets():
            y_true = training_set.convert_targets_to_dummy_matrix()
        else:
            y_true = training_set.get_targets()

        r2_list = []
        pls = self.get_result()
        ncomp = pls.x_scores_.shape[1]
        for i in range(0, ncomp):
            y_pred = np.dot(
                pls.x_scores_[:, i].reshape(-1, 1),
                pls.y_loadings_[:, i].reshape(-1, 1).T)
            y_pred = DataFrame(y_pred)
            std = y_true.std(axis=0, ddof=1)
            mean = y_true.mean(axis=0)

            print(y_pred)

            for k in range(0, y_true.shape[1]):
                y_pred.iloc[:, k] = y_pred.iloc[:, k] * std.iat[k] + mean.iat[k]
                # y_pred = y_pred*y_true.std(axis=0, ddof=1) + y_true.mean(axis=0)  # reverse normalize

            r2_list.append(r2_score(y_true, y_pred))

        index = [f"PC{n+1}" for n in range(0, ncomp)]
        columns = ["ExplainedVariance"]
        data = DataFrame(r2_list, columns=columns, index=index)
        table = Table(data=data)
        table.name = self.VARIANCE_TABLE_NAME
        self.add_resource(table)
        self.add_technical_info(TechnicalInfo(key='PC1', value=f'{data.iat[0,0]:.3f}'))
        self.add_technical_info(TechnicalInfo(key='PC2', value=f'{data.iat[1,0]:.3f}'))

    @ view(view_type=ScatterPlot2DView, human_name='2D-score plot', short_description='2D-score plot')
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
        var = self.get_variance_table().get_data()
        _view.x_label = f'PC1 ({100*var.iat[0,0]:.2f}%)'
        _view.y_label = f'PC2 ({100*var.iat[1,0]:.2f}%)'
        return _view

    @ view(view_type=ScatterPlot2DView, human_name='Prediction plot', short_description='Prediction plot')
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


@ task_decorator("PLSTrainer", human_name="PLS trainer",
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
            y = dataset.convert_targets_to_dummy_matrix().values
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


@ task_decorator("PLSTransformer", human_name="PLS transformer",
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


@ task_decorator("PLSPredictor", human_name="PLS predictor",
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

        training_set = learned_model.get_training_set()
        if training_set.has_string_targets():
            y_true = training_set.convert_targets_to_dummy_matrix()
        else:
            y_true = training_set.get_targets()

        result_dataset = Dataset(
            data=Y,
            row_names=dataset.row_names,
            column_names=list(y_true.columns),
            target_names=list(y_true.columns)
        )
        return {'result': result_dataset}
