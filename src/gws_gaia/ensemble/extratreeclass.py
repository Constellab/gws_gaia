# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, IntParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from numpy import ravel
from sklearn.ensemble import ExtraTreesClassifier

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# ExtraTreesClassifierResult
#
# *****************************************************************************


@resource_decorator("ExtraTreesClassifierResult", hide=True)
class ExtraTreesClassifierResult(BaseResource):
    pass

# *****************************************************************************
#
# ExtraTreesClassifierTrainer
#
# *****************************************************************************


@task_decorator("ExtraTreesClassifierTrainer", human_name="Extra-Trees classifier trainer",
                short_description="Train an extra-trees classifier model")
class ExtraTreesClassifierTrainer(Task):
    """
    Trainer of an extra-trees classifier. Build a forest of trees from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier.fit for more details
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': ExtraTreesClassifierResult}
    config_specs = {
        'nb_estimators': IntParam(default_value=100, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        etc = ExtraTreesClassifier(n_estimators=params["nb_estimators"])
        etc.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = ExtraTreesClassifierResult(training_set=dataset, result=etc)
        return {'result': result}

# *****************************************************************************
#
# ExtraTreesClassifierPredictor
#
# *****************************************************************************


@task_decorator("ExtraTreesClassifierPredictor", human_name="Extra-Trees classifier predictor",
                short_description="Predict dataset labels using a trained extra-trees classifier model")
class ExtraTreesClassifierPredictor(Task):
    """
    Predictor of an extra-trees classifier. Predict class for a dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier.fit for more details
    """
    input_specs = {'dataset': Dataset, 'learned_model': ExtraTreesClassifierResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        etc = learned_model.get_result()
        y = etc.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
