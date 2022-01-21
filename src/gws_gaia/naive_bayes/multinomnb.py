# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from gws_core import Dataset
from ..base.base_resource import BaseResource

# *****************************************************************************
#
# MultinomialNaiveBayesClassifierResult
#
# *****************************************************************************

@resource_decorator("MultinomialNaiveBayesClassifierResult", hide=True)
class MultinomialNaiveBayesClassifierResult(BaseResource):
    pass

# *****************************************************************************
#
# MultinomialNaiveBayesClassifierTrainer
#
# *****************************************************************************

@task_decorator("MultinomialNaiveBayesClassifierTrainer", human_name="MNB trainer",
                short_description="Predict the class labels using a Multinomial Naive Bayes (MNB) classifier")
class MultinomialNaiveBayesClassifierTrainer(Task):
    """
    Trainer of a naive Bayes classifier for a multinomial model. Fit a naive Bayes classifier according to a training dataset.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html for more details.
    """
    input_specs = {'dataset' : Dataset}
    output_specs = {'result' : MultinomialNaiveBayesClassifierResult}
    config_specs = {
        'alpha':FloatParam(default_value=1)
    }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        mnb = MultinomialNB(alpha=params["alpha"])
        mnb.fit(dataset.get_features().values, ravel(dataset.get_targets().values))
        result = MultinomialNaiveBayesClassifierResult(result = mnb)
        return {'result': result}

# *****************************************************************************
#
# MultinomialNaiveBayesClassifierPredictor
#
# *****************************************************************************

@task_decorator("MultinomialNaiveBayesClassifierPredictor", human_name="MNB classifier predictor",
                short_description="Predict the class labels using Multinomial Naive Bayes (MNB) classifier")
class MultinomialNaiveBayesClassifierPredictor(Task):
    """
    Predictor of a naïve Bayes classifier for a multinomial model. Predict class labels for a dataset using a trained naïve Bayes classifier.

    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html for more details.
    """
    input_specs = {'dataset' : Dataset, 'learned_model': MultinomialNaiveBayesClassifierResult}
    output_specs = {'result' : Dataset}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        mnb = learned_model.result
        y = mnb.predict(dataset.get_features().values)
        result_dataset = Dataset(
            data=y,
            row_names=dataset.row_names,
            column_names=dataset.target_names,
            target_names=dataset.target_names
        )
        return {'result': result_dataset}