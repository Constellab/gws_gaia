from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner)
from gws_gaia import SGDClassifierPredictor, SGDClassifierTrainer
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Linear classifier with stochastic gradient descent (SGD)")
        dataset = GWSGaiaTestHelper.get_dataset(index=5, header=0, targets=['target'])

        # run trainer
        tester = TaskRunner(
            params={'alpha': 0.01},
            inputs={'dataset': dataset},
            task_type=SGDClassifierTrainer
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
            task_type=SGDClassifierPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
