
import os
import asyncio


from gws_gaia import Dataset
from gws_gaia import LogisticRegressionTrainer, LogisticRegressionPredictor, LogisticRegressionTester
from gws_core import Settings, GTest, BaseTestCase, TaskTester, File, ConfigParams

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Logistic regression classifier")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./iris.csv")), 
            ConfigParams({
                "delimiter":",", 
                "header":0, 
                "targets":['variety']
            })
        )

        # run trainer
        tester = TaskTester(
            params = {'inv_reg_strength': 1e5},
            inputs = {'dataset': dataset},
            task_type = LogisticRegressionTrainer
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
            task_type = LogisticRegressionPredictor
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
            task_type = LogisticRegressionTester
        )
        outputs = await tester.run()
        tester_result = outputs['result']
       
        print(trainer_result)
        print(predictor_result)
        print(tester_result)