import os
import asyncio


from gws_core import Settings, BaseTestCase, TaskTester, File, ConfigParams
from gws_gaia import Dataset, DatasetImporter
from gws_gaia import AdaBoostRegressorTrainer, AdaBoostRegressorPredictor, AdaBoostRegressorTester

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("AdaBoost regressor")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")
        
        #import data
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./dataset2.csv")), 
            ConfigParams({
                "delimiter":",", 
                "header":0, 
                "targets":['target']
            })
        )
        
        # run trainer
        tester = TaskTester(
            params = {'nb_estimators': 30},
            inputs = {'dataset': dataset},
            task_type = AdaBoostRegressorTrainer
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
            task_type = AdaBoostRegressorPredictor
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
            task_type = AdaBoostRegressorTester
        )
        outputs = await tester.run()
        tester_result = outputs['result']

        print(trainer_result)
        print(tester_result)
        print(predictor_result)