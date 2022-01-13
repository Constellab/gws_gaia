# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel, unique, shape
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam,
                        StrParam, ScatterPlot2DView, ScatterPlot3DView, TableView, BarPlotView,
                        view, ResourceRField, FloatRField, IntRField, Table)
from gws_core import Dataset
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# LDAResult
#
# *****************************************************************************

@resource_decorator("LDAResult", 
                    human_name="LDA Result", 
                    short_description = "Linear Discriminant Analysis result", 
                    hide=True)
class LDAResult(BaseResource):

    _training_set: Resource = ResourceRField()
    _nb_components: int = IntRField()

    def _get_transformed_data(self) -> DataFrame: 
        lda: LinearDiscriminantAnalysis = self.get_result() 
        ncomp = self._nb_components
        X_transformed: DataFrame = lda.transform(self._training_set.get_features().values)
        columns = [f"PC{i+1}" for i in range(0,ncomp)]
        X_transformed = DataFrame(
            data=X_transformed, 
            columns=columns, 
            index=self._training_set.row_names
        )
        return X_transformed

    @view(view_type=TableView, human_name="ProjectedDataTable' table", short_description="Table of data in the score plot")
    def view_transformed_data_as_table(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """
    
        x_transformed = self._get_transformed_data() 
        table = Table(data=x_transformed)
        return TableView(table)

    @view(view_type=TableView, human_name="VarianceTable", short_description="Table of explained variances")
    def view_variance_as_table(self, params: ConfigParams) -> dict:
        """
        View table data
        """

        lda = self.get_result()
        ncomp = self._nb_components
        index = [f"PC{i+1}" for i in range(0,ncomp)]
        columns = ["ExplainedVariance"]
        data = DataFrame(lda.explained_variance_ratio_, index=index, columns=columns)
        table = Table(data)
        return TableView(table)

    @view(view_type=BarPlotView, human_name="VarianceBarPlot", short_description="Barplot of explained variances")
    def view_variance_as_barplot(self, params: ConfigParams) -> dict:
        """
        View bar plot of explained variances
        """

        lda = self.get_result()
        ncomp = self._nb_components
        explained_var: DataFrame = lda.explained_variance_ratio_
        columns = [f"PC{n+1}" for n in range(0, ncomp)]
        _view = BarPlotView()
        _view.add_series(
            x=columns,
            y=explained_var.tolist()
        )
        _view.x_label = 'Principal components'
        _view.y_label = 'Explained variance'

        return _view

    @view(view_type=ScatterPlot2DView, human_name='ScorePlot2D', short_description='2D score plot')
    def view_scores_as_2d_plot(self, params: ConfigParams) -> dict:
        """
        View 2D score plot
        """

        data = self._get_transformed_data()
        _view = ScatterPlot2DView()
        _view.add_series(
            x=data['PC1'].to_list(),
            y=data['PC2'].to_list()
        )
        _view.x_label = 'PC1'
        _view.y_label = 'PC2'
        return _view

        # data = self._get_transformed_data()
        # data_x = data['PC1'].to_list()
        # data_y = data['PC2'].to_list()

        # labels = self._training_set.get_targets().values
        # label_values = unique(labels)
        # flatten_labels = labels.flatten()
        # print(shape(labels))
        
        # _view = ScatterPlot2DView()        
        # for l in label_values:
        #     extr_data_x = data_x[flatten_labels==l]   
        #     extr_data_y = data_y[flatten_labels==l]               
        #     _view.add_series(
        #         x = extr_data_x,
        #         y = extr_data_y
        #     )
        
        # _view.x_label = 'PC1'
        # _view.y_label = 'PC2'
        # return _view

    @view(view_type=ScatterPlot3DView, human_name='ScorePlot3D', short_description='3D score plot')
    def view_scores_as_3d_plot(self, params: ConfigParams) -> dict:
        """
        View 3D score plot
        """

        data = self._get_transformed_data()
        _view = ScatterPlot2DView()
        _view.add_series(
            x=data['PC1'].to_list(),
            y=data['PC2'].to_list(),
            z=data['PC3'].to_list()
        )
        _view.x_label = 'PC1'
        _view.y_label = 'PC2'
        _view.z_label = 'PC3'
        return _view

# *****************************************************************************
#
# LDATrainer
#
# *****************************************************************************

@task_decorator("LDATrainer")
class LDATrainer(Task):
    """
    Trainer of a linear discriminant analysis classifier. Fit Linear Discriminant Analysis model according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : LDAResult}
    config_specs = {
        'solver': StrParam(default_value='svd'),
        'nb_components': IntParam(default_value=None, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        lda = LinearDiscriminantAnalysis(solver=params["solver"],n_components=params["nb_components"])
        lda.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = LDAResult(result = lda)
        result._training_set = dataset
        result._nb_components = params['nb_components']
        return {'result': result}

# *****************************************************************************
#
# LDATransformer
#
# *****************************************************************************

@task_decorator("LDATransformer")
class LDATransformer(Task):
    """
    Transformer of a linear discriminant analysis classifier. Project data to maximize class separation.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details

    """
    input_specs = {'dataset' : Dataset, 'learned_model': LDAResult}
    output_specs = {'result' : Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lda = learned_model.result
        x = lda.transform(dataset.get_features().values)
        ncomp = x.shape[1]
        result_dataset = Dataset(
            data = DataFrame(x),
            row_names = dataset.row_names,
            column_names = [f"PC{i+1}" for i in range(0,ncomp)]
        )
        return {'result': result_dataset}

# *****************************************************************************
#
# LDAPredictor
#
# *****************************************************************************

@task_decorator("LDAPredictor")
class LDAPredictor(Task):
    """
    Predictor of a linear discriminant analysis classifier. Predict class labels for samples in a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': LDAResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        lda = learned_model.result
        y = lda.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data = DataFrame(y),
            row_names = dataset.row_names,
            column_names = dataset.target_names,
            target_names = dataset.target_names,
        )
        return {'result': result_dataset}