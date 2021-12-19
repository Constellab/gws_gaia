from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner)
from gws_gaia import ExtraTreesRegressorPredictor, ExtraTreesRegressorTrainer
from gws_gaia.extra import DataProvider


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Extratrees Regressor")
        dataset = DataProvider.get_diabetes_dataset()

        # run trainer
        tester = TaskRunner(
            params={'nb_estimators': 100},
            inputs={'dataset': dataset},
            task_type=ExtraTreesRegressorTrainer
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
            task_type=ExtraTreesRegressorPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
