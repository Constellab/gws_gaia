# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from gws_core import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("KNNClassifierResult", hide=True)
class KNNClassifierResult(BaseResource):
    pass
    # _training_set: Resource = ResourceRField()

    # def _get_predicted_data(self) -> DataFrame:
    #     neigh: KNeighborsClassifier = self.get_result() #lir du type Linear Regression
    #     Y_predicted: DataFrame = neigh.predict(self._training_set.get_features().values)
    #     Y_predicted = DataFrame(data=Y_predicted)
    #     return Y_predicted    

#==============================================================================
#==============================================================================

@task_decorator("KNNClassifierTrainer")
class KNNClassifierTrainer(Task):
    """
    Trainer of a k-nearest neighbors classifier. Fit the k-nearest neighbors classifier from the training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : KNNClassifierResult}
    config_specs = {
        'nb_neighbors': IntParam(default_value=5, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        neigh = KNeighborsClassifier(n_neighbors=params["nb_neighbors"])
        neigh.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = KNNClassifierResult(result = neigh)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("KNNClassifierPredictor")
class KNNClassifierPredictor(Task):
    """
    Predictor of a K-nearest neighbors classifier. Predict the class labels for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': KNNClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        neigh = learned_model.result
        y = neigh.predict(dataset.get_features().values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}