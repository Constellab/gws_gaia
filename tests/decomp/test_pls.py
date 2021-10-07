
import os
import asyncio


from gws_gaia import Dataset, DatasetLoader
from gws_gaia import PLSTrainer, PLSPredictor, PLSTester
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Partial Least Squares (PLS) regression")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(os.path.join(
            test_dir, "./dataset1.csv"), 
            delimiter=",", 
            header=0, 
            targets=['target1', 'target2']
        )

        # run trainer
        tester = TaskTester(
            params = {'nb_components': 2},
            inputs = {'dataset': dataset},
            task_type = PLSTrainer
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
            task_type = PLSPredictor
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
            task_type = PLSTester
        )
        outputs = await tester.run()
        tester_result = outputs['result']
       
        print(trainer_result)
        print(predictor_result)
        print(tester_result)
