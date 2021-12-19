from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner)
from gws_gaia import QDAPredictor, QDATrainer
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Quadratic discriminant analysis")
        dataset = GWSGaiaTestHelper.get_dataset(index=3, header=0, targets=['target'])

        # run trainer
        tester = TaskRunner(
            params={'reg_param': 0},
            inputs={'dataset': dataset},
            task_type=QDATrainer
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
            task_type=QDAPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
