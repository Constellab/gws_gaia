from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner)
from gws_gaia import (MultinomialNaiveBayesClassifierPredictor,
                      MultinomialNaiveBayesClassifierTrainer)
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Naive Bayes classifier for a multinomial model")
        dataset = GWSGaiaTestHelper.get_dataset(index=7, header=0, targets=['target'])

        # run trainer
        tester = TaskRunner(
            params={'alpha': 1},
            inputs={'dataset': dataset},
            task_type=MultinomialNaiveBayesClassifierTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskRunner(
            params={},
            inputs={
                'dataset': dataset,
                'learned_model': trainer_result
            },
            task_type=MultinomialNaiveBayesClassifierPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
