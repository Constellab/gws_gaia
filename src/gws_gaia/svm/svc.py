# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from gws_core import (BoolParam, ConfigParams, Dataset, FloatParam, IntParam,
                      Resource, StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from pandas import DataFrame
from sklearn.svm import SVC

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# SVCResult
#
# *****************************************************************************


@resource_decorator("SVCResult", hide=True)
class SVCResult(BaseResourceSet):
    pass

# *****************************************************************************
#
# SVCTrainer
#
# *****************************************************************************


@task_decorator("SVCTrainer", human_name="SVC trainer",
                short_description="Train a C-Support Vector Classifier (SVC) model")
class SVCTrainer(Task):
    """
    Trainer of a C-Support Vector Classifier (SVC) model. Fit a SVC model according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(SVCResult, human_name="result", short_description="The output result")}
    config_specs = {
        'probability': BoolParam(default_value=False),
        'kernel': StrParam(default_value='rbf')
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        svc = SVC(probability=params["probability"], kernel=params["kernel"])
        svc.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = SVCResult(training_set=dataset, result=svc)
        return {'result': result}

# *****************************************************************************
#
# SVCPredictor
#
# *****************************************************************************


@task_decorator("SVCPredictor", human_name="SVC predictor",
                short_description="Predict dataset class labels using a trained C-Support Vector Classifier (SVC) model")
class SVCPredictor(Task):
    """
    Predictor of a C-Support Vector Classifier (SVC) model. Predict class labels of a dataset with a trained SVC model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(SVCResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        svc = learned_model.get_result()
        y = svc.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
