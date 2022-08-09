# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from gws_core import (ConfigParams, Dataset, FloatParam, IntParam, Resource,
                      StrParam, Task, TaskInputs, TaskOutputs,
                      resource_decorator, task_decorator, InputSpec, OutputSpec)
from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import SGDClassifier

from ..base.base_resource import BaseResourceSet

# *****************************************************************************
#
# SGDClassifierResult
#
# *****************************************************************************

@resource_decorator("SGDClassifierResult")
class SGDClassifierResult(BaseResourceSet):
    pass

# *****************************************************************************
#
# SGDClassifierTrainer
#
# *****************************************************************************

@task_decorator("SGDClassifierTrainer", human_name="SGD classifier trainer",
                short_description="Train a stochastic gradient descent (SGD) linear classifier")
class SGDClassifierTrainer(Task):
    """
    Trainer of a linear classifier with stochastic gradient descent (SGD). Fit a SGD linear classifier with a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset")}
    output_specs = {'result': OutputSpec(SGDClassifierResult, human_name="result", short_description="The output result")}
    config_specs = {
        'loss':StrParam(default_value='hinge'),
        'alpha': FloatParam(default_value=0.0001, min_value=0),
        'max_iter':IntParam(default_value=1000, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        sgdc = SGDClassifier(max_iter=params["max_iter"],alpha=params["alpha"],loss=params["loss"])
        sgdc.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = SGDClassifierResult(training_set=dataset,result = sgdc)
        return {'result': result}

# *****************************************************************************
#
# SGDClassifierPredictor
#
# *****************************************************************************

@task_decorator("SGDClassifierPredictor", human_name="SGD classifier predictor",
                short_description="Predict class labels using a trained SGD classifier")
class SGDClassifierPredictor(Task):
    """
    Predictor of a linear classifier with stochastic gradient descent (SGD). Predict class labels of a dataset with a trained SGD linear classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html for more details.
    """
    input_specs = {'dataset': InputSpec(Dataset, human_name="Dataset", short_description="The input dataset"),
            'learned_model': InputSpec(SGDClassifierResult, human_name="Learned model", short_description="The input model")}
    output_specs = {'result': OutputSpec(Dataset, human_name="result", short_description="The output result")}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        sgdc = learned_model.get_result()
        y = sgdc.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}
