# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws_core import (ConfigParams, Dataset, FloatParam, StrParam, Task,
                      TaskInputs, TaskOutputs, resource_decorator,
                      task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from sklearn.kernel_ridge import KernelRidge

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# KernelRidgeResult
#
# *****************************************************************************


@resource_decorator("KernelRidgeResult", hide=True)
class KernelRidgeResult(BaseResourceSet):
    pass

# *****************************************************************************
#
# KernelRidgeTrainer
#
# *****************************************************************************


@task_decorator("KernelRidgeTrainer", human_name="Kernel-Ridge trainer",
                short_description="Train a kernel ridge regression model")
class KernelRidgeTrainer(Task):
    """
    Trainer of a kernel ridge regression model. Fit a kernel ridge regression model with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(KernelRidgeResult, human_name="result", short_description="The output result")}
    config_specs = {
        'gamma': FloatParam(default_value=None),
        'kernel': StrParam(default_value='linear')
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        krr = KernelRidge(gamma=params["gamma"], kernel=params["kernel"])
        krr.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = KernelRidgeResult(training_set=dataset, result=krr)
        return {'result': result}

# *****************************************************************************
#
# KernelRidgePredictor
#
# *****************************************************************************


@task_decorator("KernelRidgePredictor", human_name="Kernel-Ridge predictor",
                short_description="Predict dataset targets using a trained kernel ridge regression model")
class KernelRidgePredictor(Task):
    """
    Predictor of a kernel ridge regression model. Predict a regression target from a dataset with a trained kernel ridge regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(KernelRidgeResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {}

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        krr = learned_model.get_result()
        y = krr.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
