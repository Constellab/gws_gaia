from gws_core import (BaseTestCase, ConfigParams, Dataset, File, Settings,
                      TaskRunner)
from gws_core.extra import DataProvider
from gws_gaia import (DecisionTreeClassifierPredictor,
                      DecisionTreeClassifierTrainer)


class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Decision Tree Classifier")
        dataset = DataProvider.get_iris_dataset()

        # run trainer
        tester = TaskRunner(
            params={'max_depth': 4},
            inputs={'dataset': dataset},
            task_type=DecisionTreeClassifierTrainer
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
            task_type=DecisionTreeClassifierPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
