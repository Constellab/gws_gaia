
import os
import asyncio


from gws_gaia import Dataset
from gws_gaia import RidgeRegressionTrainer, RidgeRegressionPredictor, RidgeRegressionTester
from gws_core import Settings, GTest, BaseTestCase, TaskTester, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Ridge regression model")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./diabetes.csv")), 
            ConfigParams({
                "delimiter":",", 
                "header":0, 
                "targets":['target']
            })
        )

        # run trainer
        tester = TaskTester(
            params = {'alpha': 1},
            inputs = {'dataset': dataset},
            task_type = RidgeRegressionTrainer
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
            task_type = RidgeRegressionPredictor
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
            task_type = RidgeRegressionTester
        )
        outputs = await tester.run()
        tester_result = outputs['result']

        print(trainer_result)
        print(tester_result)
        print(predictor_result)
