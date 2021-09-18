# LICENSE
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from numpy import ravel
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB

from gws_core import (Task, Resource, task_decorator, resource_decorator,
                        ConfigParams, TaskInputs, TaskOutputs, IntParam, FloatParam, StrParam)

from ..data.core import GenericResult
from ..data.dataset import Dataset
from ..base.base_resource import BaseResource

#==============================================================================
#==============================================================================

@resource_decorator("MultinomialNaiveBayesClassifierResult", hide=True)
class MultinomialNaiveBayesClassifierResult(BaseResource):
    pass

#==============================================================================
#==============================================================================

@task_decorator("MultinomialNaiveBayesClassifierTrainer")
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
        mnb.fit(dataset.features.values, ravel(dataset.targets.values))
        result = MultinomialNaiveBayesClassifierResult.from_result(result=mnb)
        return {'result': result}

#==============================================================================
#==============================================================================

@task_decorator("MultinomialNaiveBayesClassifierTester")
class MultinomialNaiveBayesClassifierTester(Task):
    """
    Tester of a naïve Bayes classifier for a multinomial model. Return the mean accuracy on a given dataset for a trained naïve Bayes classifier for a multinomial model.
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html for more details
    """
    input_specs = {'dataset' : Dataset, 'learned_model': MultinomialNaiveBayesClassifierResult}
    output_specs = {'result' : GenericResult}
    config_specs = {   }

    async def run(self, params: ConfigParams, inputs: TaskInputs) -> TaskOutputs:
        dataset = inputs['dataset']
        learned_model = inputs['learned_model']
        mnb = learned_model.binary_store['result']
        y = mnb.score(dataset.features.values, dataset.targets.values)
        z = tuple([y])
        result_dataset = GenericResult.from_result(result = z)
        return {'result': result_dataset}

#==============================================================================
#==============================================================================

@task_decorator("MultinomialNaiveBayesClassifierPredictor")
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
        mnb = learned_model.binary_store['result']
        y = mnb.predict(dataset.features.values)
        result_dataset = Dataset(targets = DataFrame(y))
        return {'result': result_dataset}