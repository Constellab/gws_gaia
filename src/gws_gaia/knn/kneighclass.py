# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, IntParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from numpy import ravel
from sklearn.neighbors import KNeighborsClassifier

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# KNNClassifierResult
#
# *****************************************************************************


@resource_decorator("KNNClassifierResult", hide=True)
class KNNClassifierResult(BaseResource):
    pass

# *****************************************************************************
#
# KNNClassifierTrainer
#
# *****************************************************************************


@task_decorator("KNNClassifierTrainer", human_name="KNN classifier trainer",
                short_description="Train a k-nearest neighbors classifier model")
class KNNClassifierTrainer(Task):
    """
    Trainer of a k-nearest neighbors classifier. Fit the k-nearest neighbors classifier from the training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html for more details.
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': KNNClassifierResult}
    config_specs = {
        'nb_neighbors': IntParam(default_value=5, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        neigh = KNeighborsClassifier(n_neighbors=params["nb_neighbors"])
        neigh.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = KNNClassifierResult(training_set=dataset, result=neigh)
        return {'result': result}

# *****************************************************************************
#
# KNNClassifierPredictor
#
# *****************************************************************************


@task_decorator("KNNClassifierPredictor", human_name="KNN classifier predictor",
                short_description="Predict dataset labels using a trained k-nearest neighbors classifier model")
class KNNClassifierPredictor(Task):
    """
    Predictor of a K-nearest neighbors classifier. Predict the class labels for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html for more details.
    """
    input_specs = {'dataset': Dataset, 'learned_model': KNNClassifierResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        neigh = learned_model.get_result()
        y = neigh.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
