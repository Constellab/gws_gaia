from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner)
from gws_gaia import GaussianNaiveBayesPredictor, GaussianNaiveBayesTrainer
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Gaussian  Naive Bayes")
        dataset = GWSGaiaTestHelper.get_dataset(index=3, header=0, targets=['target'])

        # run trainer
        tester = TaskRunner(
            params={},
            inputs={'dataset': dataset},
            task_type=GaussianNaiveBayesTrainer
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
            task_type=GaussianNaiveBayesPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
