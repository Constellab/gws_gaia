# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.linear_model import ElasticNet

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)
from gws_core import Dataset
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# ElasticNetResult
#
# *****************************************************************************

@resource_decorator("ElasticNetResult", hide=True)
class ElasticNetResult(BaseResource):
    pass

# *****************************************************************************
#
# ElasticNetTrainer
#
# *****************************************************************************

@task_decorator("ElasticNetTrainer")
class ElasticNetTrainer(Task):
    """ 
    Trainer of an elastic net model. Fit model with coordinate descent.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for more details
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : ElasticNetResult}
    config_specs = {
        'alpha': FloatParam(default_value=1, min_value=0)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        eln = ElasticNet(alpha=params["alpha"])
        eln.fit(dataset.get_features().values, dataset.get_targets().values)
        result = ElasticNetResult(result = eln)
        return {'result': result}

# *****************************************************************************
#
# ElasticNetPredictor
#
# *****************************************************************************

@task_decorator("ElasticNetPredictor")
class ElasticNetPredictor(Task):
    """
    Predictor of a trained elastic net model. Predict from a dataset using the trained model.

    See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': ElasticNetResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        eln = learned_model.result
        y = eln.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y, 
            row_names=dataset.row_names, 
            column_names=dataset.target_names, 
            target_names=dataset.target_names
        )
        return {'result': result_dataset}