
import os
import asyncio


from gws_gaia import Dataset, DatasetLoader
from gws_gaia import SVCTrainer, SVCPredictor, SVCTester
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("C-Support Vector Classifier (SVC)")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(os.path.join(
            test_dir, "./iris.csv"), 
            delimiter=",", 
            header=0, 
            targets=['variety']
        )

        # run trainer
        tester = TaskTester(
            params = {'probability': True},
            inputs = {'dataset': dataset},
            task_type = SVCTrainer
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
            task_type = SVCPredictor
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
            task_type = SVCTester
        )
        outputs = await tester.run()
        tester_result = outputs['result']

        print(trainer_result)
        print(tester_result)
        print(predictor_result)
