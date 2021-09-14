import os
import asyncio


from gws_core import Settings, GTest, BaseTestCase, TaskTester, TaskInputs, ConfigParams
from gws_gaia import Dataset, DatasetLoader
from gws_gaia import AdaBoostRegressorTrainer, AdaBoostRegressorPredictor, AdaBoostRegressorTester

class TestTrainer(BaseTestCase):

    async def test_process(self):
        GTest.print("AdaBoost regressor")
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
            params = ConfigParams({'nb_estimators': 30}),
            inputs = TaskInputs({'dataset': dataset}),
            task = AdaBoostRegressorTrainer()
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskTester(
            params = ConfigParams(),
            inputs = TaskInputs({
                'dataset': dataset, 
                'learned_model': trainer_result
            }),
            task = AdaBoostRegressorPredictor()
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        # run tester
        tester = TaskTester(
            params = ConfigParams(),
            inputs = TaskInputs({
                'dataset': dataset, 
                'learned_model': trainer_result
            }),
            task = AdaBoostRegressorTester()
        )
        outputs = await tester.run()
        tester_result = outputs['result']

        print(trainer_result)
        print(tester_result)
        print(predictor_result)
