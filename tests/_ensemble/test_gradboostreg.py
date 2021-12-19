from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner)
from gws_gaia import (GradientBoostingRegressorPredictor,
                      GradientBoostingRegressorTrainer)
from gws_gaia.extra import DataProvider


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Gradient Boosting Regressor")
        dataset = DataProvider.get_diabetes_dataset()

        # run trainer
        tester = TaskRunner(
            params={},
            inputs={'dataset': dataset},
            task_type=GradientBoostingRegressorTrainer
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
            task_type=GradientBoostingRegressorPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
