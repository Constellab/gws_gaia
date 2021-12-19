from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner)
from gws_core.extra import DataProvider
from gws_gaia import RidgeClassifierPredictor, RidgeClassifierTrainer


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Ridge classifier model")
        dataset = DataProvider.get_iris_dataset()

        # run trainer
        tester = TaskRunner(
            params={'alpha': 1},
            inputs={'dataset': dataset},
            task_type=RidgeClassifierTrainer
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
            task_type=RidgeClassifierPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
