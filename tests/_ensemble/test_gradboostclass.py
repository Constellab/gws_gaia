from gws_core import (BaseTestCase, ConfigParams, Dataset, File, GTest,
                      Settings, TaskRunner)
from gws_core.extra import DataProvider
from gws_gaia import (GradientBoostingClassifierPredictor,
                      GradientBoostingClassifierTrainer)


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Gradient Boosting Classifier")
        dataset = DataProvider.get_iris_dataset()

        # run trainer
        tester = TaskRunner(
            params={'nb_estimators': 25},
            inputs={'dataset': dataset},
            task_type=GradientBoostingClassifierTrainer
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
            task_type=GradientBoostingClassifierPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
