# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com


from pandas import DataFrame
from sklearn.cross_decomposition import PLSRegression

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("PLSResult", hide=True)
class PLSResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("PLSTrainer")
class PLSTrainer(Task):
    """
    Trainer of a Partial Least Squares (PLS) regression model. Fit a PLS regression model to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : PLSResult}
    config_specs = {
        'nb_components': IntParam(default_value=2, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        pls = PLSRegression(n_components=params["nb_components"])
        pls.fit(dataset.features.values, dataset.targets.values)
        result = PLSResult(result = pls)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("PLSTester")
class PLSTester(Task):
    """
    Tester of a trained Partial Least Squares (PLS) regression model. Return the coefficient of determination R^2 of the prediction on a given dataset for a trained Partial Least Squares (PLS) regression model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': PLSResult}
    output_specs = {'result' : GenericResult}
    config_specs = {  }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        pls = learned_model.result
        y = pls.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = GenericResult(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("PLSPredictor")
class PLSPredictor(Task):
    """
    Predictor of a Partial Least Squares (PLS) regression model. Predict targets of a dataset with a trained PLS regression model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': PLSResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        pls = learned_model.result
        y = pls.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}