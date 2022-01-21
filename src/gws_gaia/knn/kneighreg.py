# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, IntParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from numpy import ravel
from sklearn.neighbors import KNeighborsRegressor

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# KNNRegressorResult
#
# *****************************************************************************


@resource_decorator("KNNRegressorResult", hide=True)
class KNNRegressorResult(BaseResource):
    pass

# *****************************************************************************
#
# KNNRegressorTrainer
#
# *****************************************************************************


@task_decorator("KNNRegressorTrainer", human_name="KNN regressor trainer",
                short_description="Train a k-nearest neighbors regression model")
class KNNRegressorTrainer(Task):
    """
    Trainer for a k-nearest neighbors regressor. Fit a k-nearest neighbors regressor from a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html for more details.
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': KNNRegressorResult}
    config_specs = {
        'nb_neighbors': IntParam(default_value=5, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        neigh = KNeighborsRegressor(n_neighbors=params["nb_neighbors"])
        neigh.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = KNNRegressorResult(result=neigh)
        return {'result': result}

# *****************************************************************************
#
# KNNRegressorPredictor
#
# *****************************************************************************


@task_decorator("KNNRegressorPredictor", human_name="KNN regressor predictor",
                short_description="Predict dataset targets using a trained k-nearest neighbors regression model")
class KNNRegressorPredictor(Task):
    """
    Predictor for a k-nearest neighbors regressor. Predict the regression target for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html for more details.
    """
    input_specs = {'dataset': Dataset, 'learned_model': KNNRegressorResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        neigh = learned_model.result
        y = neigh.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
