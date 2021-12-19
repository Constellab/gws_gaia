from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner)
from gws_gaia import (GaussianProcessRegressorPredictor,
                      GaussianProcessRegressorTrainer)
from gws_gaia.extra import DataProvider


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Gaussian Process Regressor")
        dataset = DataProvider.get_diabetes_dataset()

        # run trainer
        tester = TaskRunner(
            params={'alpha': 1e-10},
            inputs={'dataset': dataset},
            task_type=GaussianProcessRegressorTrainer
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
            task_type=GaussianProcessRegressorPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
