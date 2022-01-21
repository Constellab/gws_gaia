# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.svm import SVR

from gws_core import (Task, Resource, resource_decorator, task_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from gws_core import Dataset
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# SVRResult
#
# *****************************************************************************

@resource_decorator("SVRResult", hide=True)
class SVRResult(BaseResource):
    pass

# *****************************************************************************
#
# SVRTrainer
#
# *****************************************************************************

@task_decorator("SVRTrainer", human_name="SVC trainer",
                short_description="Train a C-Support Vector Regression (SVC) model")
class SVRTrainer(Task):
    """
    Trainer of a Epsilon-Support Vector Regression (SVR) model. Fit a SVR model according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : SVRResult}
    config_specs = {
        'kernel':StrParam(default_value='rbf')
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        svr = SVR(kernel=params["kernel"])
        svr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = SVRResult(result=svr)
        return {'result': result}

# *****************************************************************************
#
# SVRPredictor
#
# *****************************************************************************

@task_decorator("SVRPredictor", human_name="SVR predictor",
                short_description="Predict dataset targets using a trained Epsilon-Support Vector Regression (SVR) model")
class SVRPredictor(Task):
    """
    Predictor of a Epsilon-Support Vector Regression (SVR) model. Predict target values of a dataset with a trained SVR model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': SVRResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        svr = learned_model.result
        y = svr.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}