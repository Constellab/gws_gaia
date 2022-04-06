# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, IntParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from numpy import ravel
from sklearn.ensemble import RandomForestClassifier

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# RandomForestClassifierResult
#
# *****************************************************************************


@resource_decorator("RandomForestClassifierResult", hide=True)
class RandomForestClassifierResult(BaseResource):
    pass

# *****************************************************************************
#
# RandomForestClassifierTrainer
#
# *****************************************************************************


@task_decorator("RandomForestClassifierTrainer", human_name="Random-Forest classifier trainer",
                short_description="Train a random forest classifier model")
class RandomForestClassifierTrainer(Task):
    """
    Trainer of a random forest classifier. Build a forest of trees from a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for more details.
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': RandomForestClassifierResult}
    config_specs = {
        'nb_estimators': IntParam(default_value=100, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        rfc = RandomForestClassifier(n_estimators=params["nb_estimators"])
        rfc.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = RandomForestClassifierResult(training_set=dataset, result=rfc)
        return {'result': result}

# *****************************************************************************
#
# RandomForestClassifierPredictor
#
# *****************************************************************************


@task_decorator("RandomForestClassifierPredictor", human_name="Random-Forest classifier predictor",
                short_description="Predict dataset labels using a trained Random forest classifier model")
class RandomForestClassifierPredictor(Task):
    """
    Predictor of a random forest classifier. Predict class labels of a dataset with a trained random forest classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html for more details.
    """
    input_specs = {'dataset': Dataset, 'learned_model': RandomForestClassifierResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        rfc = learned_model.get_result()
        y = rfc.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
