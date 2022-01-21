# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, IntParam, Task, TaskInputs,
                      TaskOutputs, resource_decorator, task_decorator)
from numpy import ravel
from sklearn.ensemble import AdaBoostClassifier

from ..base.base_resource import BaseResource

# *****************************************************************************
#
# AdaBoostClassifierResult
#
# *****************************************************************************


@resource_decorator("AdaBoostClassifierResult", hide=True)
class AdaBoostClassifierResult(BaseResource):
    """AdaBoostClassifierResult"""
    pass

# *****************************************************************************
#
# AdaBoostClassifierTrainer
#
# *****************************************************************************


@task_decorator("AdaBoostClassifierTrainer", human_name="AdaBoost classifier trainer",
                short_description="Train an AdaBoost classifier model")
class AdaBoostClassifierTrainer(Task):
    """
    Trainer of an AdaBoost classifier. This process builds a boosted classifier from a training set.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html for more details
    """
    input_specs = {'dataset': Dataset}
    output_specs = {'result': AdaBoostClassifierResult}
    config_specs = {
        'nb_estimators': IntParam(default_value=50, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        abc = AdaBoostClassifier(n_estimators=params["nb_estimators"])
        abc.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = AdaBoostClassifierResult(result=abc)
        return {"result": result}

# *****************************************************************************
#
# AdaBoostClassifierPredictor
#
# *****************************************************************************


@task_decorator("AdaBoostClassifierPredictor", human_name="AdaBoost classifier predictor",
                short_description="Predict dataset labels using a trained AdaBoost classifier model")
class AdaBoostClassifierPredictor(Task):
    """
    Predictor of a trained AdaBoost classifier. This process predicts classes for a dataset.
    The predicted class of an input sample is computed as the weighted mean prediction of the classifiers in the ensemble.

    See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html for more details
    """
    input_specs = {'dataset': Dataset, 'learned_model': AdaBoostClassifierResult}
    output_specs = {'result': Dataset}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        abc = learned_model.result
        y = abc.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
