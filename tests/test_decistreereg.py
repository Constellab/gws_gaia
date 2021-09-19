
import os
import asyncio


from gws_gaia import Dataset, DatasetLoader
from gws_gaia import DecisionTreeRegressorTrainer, DecisionTreeRegressorPredictor, DecisionTreeRegressorTester 
from gws_core import Settings, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Decision Tree Regressor")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(os.path.join(
            test_dir, "./dataset2.csv"), 
            delimiter=",", 
            header=0, 
            targets=['target']
        )

        # run trainer
        tester = TaskTester(
            params = {'max_depth': 4},
            inputs = {'dataset': dataset},
            task_type = DecisionTreeRegressorTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskTester(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = DecisionTreeRegressorPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        # run tester
        tester = TaskTester(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = DecisionTreeRegressorTester
        )
        outputs = await tester.run()
        tester_result = outputs['result']

        print(trainer_result)
        print(tester_result)
        print(predictor_result)
