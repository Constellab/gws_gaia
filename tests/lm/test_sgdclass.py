
import os
import asyncio


from gws_gaia import Dataset, DatasetLoader
from gws_gaia import SGDClassifierTrainer, SGDClassifierPredictor, SGDClassifierTester
from gws_core import Settings, GTest, BaseTestCase, TaskTester

class TestTrainer(BaseTestCase):

    async def test_process(self):
        self.print("Linear classifier with stochastic gradient descent (SGD)")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(os.path.join(
            test_dir, "./dataset5.csv"), 
            delimiter=",", 
            header=0, 
            targets=['target']
        )

        # run trainer
        tester = TaskTester(
            params = {'alpha': 0.01},
            inputs = {'dataset': dataset},
            task_type = SGDClassifierTrainer
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
            task_type = SGDClassifierPredictor
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
            task_type = SGDClassifierTester
        )
        outputs = await tester.run()
        tester_result = outputs['result']

        print(trainer_result)
        print(tester_result)
        print(predictor_result)
